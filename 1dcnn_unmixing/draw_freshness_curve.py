import os
import glob
import argparse
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Unmix1DCNNAE
from load_endmember import load_endmembers_from_folder


# ----------------------------
# 时间点
# ----------------------------
TIMEPOINTS = [
    ("T1",  "2025/12/23 11:52"),
    ("T2",  "2025/12/23 14:08"),
    ("T3",  "2025/12/23 15:36"),
    ("T4",  "2025/12/23 16:28"),
    ("T5",  "2025/12/23 17:31"),
    ("T6",  "2025/12/23 19:09"),
    ("T8",  "2025/12/23 20:18"),
    ("T9",  "2025/12/23 21:32"),
    ("T10", "2025/12/23 22:28"),
    ("T11", "2025/12/23 22:54"),
    ("T12", "2025/12/24 09:55"),
    ("T13", "2025/12/24 12:58"),
    ("T14", "2025/12/24 15:05"),
    ("T15", "2025/12/24 16:48"),
    ("T16", "2025/12/24 19:05"),
    ("T17", "2025/12/24 21:47"),
    ("T18", "2025/12/24 23:10"),
    ("T19", "2025/12/25 09:50"),
]


def parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y/%m/%d %H:%M")


def load_mat_cube(path: str, key: str = "hyper_image"):
    """
    Load v7.3 mat cube: (bands, samples, lines)
    Return:
        x2d_np: (samples*lines, bands) float32
        H, W
    """
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"[{os.path.basename(path)}] key '{key}' not found. Available keys: {list(f.keys())}")
        cube = np.array(f[key])  # (bands, samples, lines)

    if cube.ndim != 3:
        raise ValueError(f"[{os.path.basename(path)}] expected 3D cube, got {cube.shape}")

    bands, samples, lines = cube.shape
    x2d_np = np.transpose(cube, (1, 2, 0)).reshape(samples * lines, bands).astype(np.float32, copy=False)
    return x2d_np, samples, lines


@torch.no_grad()
def infer_abundance_in_chunks(model: torch.nn.Module,
                              x2d_np: np.ndarray,
                              device: torch.device,
                              infer_bs: int = 8192) -> torch.Tensor:
    """
    Chunk inference to avoid OOM.
    Args:
        x2d_np: (N, bands) float32 numpy
    Returns:
        a_cpu: (N, K) float32 torch on CPU
    """
    model.eval()
    N = x2d_np.shape[0]
    a_list = []
    for s in range(0, N, infer_bs):
        e = min(s + infer_bs, N)
        xb = torch.from_numpy(x2d_np[s:e]).to(device, non_blocking=True)  # (bs, bands)
        _, a = model(xb)  # (bs, K)
        a_list.append(a.detach().cpu())
        del xb, a
    return torch.cat(a_list, dim=0)


def compute_freshness_index(a_cpu: torch.Tensor, eps: float = 1e-12) -> float:
    """
    a_cpu: (N,5) abundances on CPU
    Order:
      0 bk
      1 fat_dry
      2 fat_fresh
      3 lean_dry
      4 lean_fresh

    Freshness index in [0,1]:
      Fresh = fat_fresh + lean_fresh
      Weight w = 1 - bk (downweight background)
      FI = sum(w*Fresh) / sum(w)
    """
    a = a_cpu.float()

    bk = a[:, 0]
    fat_dry = a[:, 1]
    fat_fresh = a[:, 2]
    lean_dry = a[:, 3]
    lean_fresh = a[:, 4]

    fresh = fat_fresh + lean_fresh
    spoiled = fat_dry + lean_dry

    w = (1.0 - bk).clamp(min=0.0, max=1.0)  # meat weight
    denom = w.sum().item() + eps

    fi = (w * fresh).sum().item() / denom  # 0~1

    # 在CSV里记录
    spoiled_ratio = (w * spoiled).sum().item() / denom
    meat_coverage = w.mean().item()  # 平均“非背景权重”，反映肉区域占比/背景遮罩情况

    return float(fi), float(spoiled_ratio), float(meat_coverage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="folder containing timepoint mat files")
    parser.add_argument("--mat_key", type=str, default="hyper_image")
    parser.add_argument("--ckpt", type=str, required=True, help="best.pth or latest.pth")
    parser.add_argument("--endmembers_dir", type=str, required=True, help="folder containing 5 endmember .mat files")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--infer_bs", type=int, default=8192, help="pixels per chunk; reduce if OOM")
    parser.add_argument("--out_dir", type=str, default="freshness_results")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load checkpoint ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    bands = int(ckpt["bands"])
    K = int(ckpt["K"])
    if K != 5:
        raise ValueError(f"This script assumes K=5, but ckpt K={K}")

    # ---- load endmembers (fixed order) ----
    E = load_endmembers_from_folder(args.endmembers_dir, bands=bands)
    if isinstance(E, tuple):
        E = E[0]  #兼容返回 (E, class_names)

    model = Unmix1DCNNAE(bands=bands, K=K, endmembers=E).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[INFO] loaded model: {args.ckpt}, bands={bands}, K={K}")

    # ---- collect mat files ----
    mat_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.mat")))
    if len(mat_paths) == 0:
        raise FileNotFoundError(f"No .mat files found in {args.test_dir}")

    # 如果不对齐，可以自己改成按文件名匹配 T1/T2...（我这里先用“顺序对齐”）
    if len(mat_paths) != len(TIMEPOINTS):
        print(f"[WARN] mat files = {len(mat_paths)} but TIMEPOINTS = {len(TIMEPOINTS)}")
        print("       This script aligns by sorted file order. Make sure they match your timepoint order.")

    records = []
    for i, p in enumerate(mat_paths):
        tp_name, tp_time = TIMEPOINTS[i] if i < len(TIMEPOINTS) else (f"T{i+1}", "N/A")
        print(f"[TEST] {tp_name}  {tp_time}  file={Path(p).name}")

        x2d_np, H, W = load_mat_cube(p, key=args.mat_key)
        a_cpu = infer_abundance_in_chunks(model, x2d_np, device=device, infer_bs=args.infer_bs)

        fi, spoiled_ratio, meat_cov = compute_freshness_index(a_cpu)

        records.append({
            "tp": tp_name,
            "datetime": tp_time,
            "file": Path(p).name,
            "freshness": fi,
            "spoiled_ratio": spoiled_ratio,
            "meat_coverage": meat_cov,
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- save CSV ----
    csv_path = out_dir / "freshness_curve.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("tp,datetime,file,freshness,spoiled_ratio,meat_coverage\n")
        for r in records:
            f.write(f"{r['tp']},{r['datetime']},{r['file']},{r['freshness']:.6f},{r['spoiled_ratio']:.6f},{r['meat_coverage']:.6f}\n")
    print(f"[OK] saved csv: {csv_path}")

    # ---- plot curve ----
    x_labels = [r["tp"] for r in records]
    y = [r["freshness"] for r in records]

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(y)), y, marker="o")
    plt.xticks(range(len(x_labels)), x_labels, rotation=0)
    plt.xlabel("Time Point")
    plt.ylabel("Freshness Index")
    plt.grid(True)
    plt.tight_layout()

    fig_path = out_dir / "freshness_curve.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[OK] saved figure: {fig_path}")

    # 画“腐败比例”曲线
    y2 = [r["spoiled_ratio"] for r in records]
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(y2)), y2, marker="o")
    plt.xticks(range(len(x_labels)), x_labels, rotation=0)
    plt.xlabel("Time Point")
    plt.ylabel("Spoiled Ratio")
    plt.grid(True)
    plt.tight_layout()

    fig2_path = out_dir / "spoiled_ratio_curve.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[OK] saved figure: {fig2_path}")


if __name__ == "__main__":
    main()