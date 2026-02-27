# test.py
import os
import glob
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Unmix1DCNNAE
from load_endmember import load_endmembers_from_folder


# =========================
# 端元颜色定义 (K=5)
# =========================
# 顺序必须和训练一致:
# 0 bk, 1 fat_dry, 2 fat_fresh, 3 lean_dry, 4 lean_fresh
COLORS = torch.tensor([
    [0.0, 0.0, 1.0],   # background (blue)
    [0.6, 0.0, 0.6],   # fat_dry (purple)
    [1.0, 1.0, 0.0],   # fat_fresh (yellow)
    [1.0, 0.0, 0.0],   # lean_dry (red)
    [0.0, 1.0, 0.0],   # lean_fresh (green)
], dtype=torch.float32)  # (5,3)


def load_mat_cube(path, key="hyper_image"):
    """
    Load v7.3 mat cube: (bands, samples, lines)
    return:
        x2d: (samples*lines, bands) float32 numpy
        H, W (samples, lines)
    """
    with h5py.File(path, "r") as f:
        cube = np.array(f[key])  # (bands, samples, lines)

    bands, samples, lines = cube.shape
    x2d = np.transpose(cube, (1, 2, 0)).reshape(samples * lines, bands)
    return x2d.astype(np.float32, copy=False), samples, lines


def infer_abundance_in_chunks(model: torch.nn.Module,
                              x2d_np: np.ndarray,
                              device: torch.device,
                              infer_bs: int = 8192) -> torch.Tensor:
    """
    按 chunk 推理，避免整幅图一次性进 GPU 造成 OOM
    Args:
        x2d_np: (N, bands) float32 numpy
    Returns:
        a_cpu: (N, K) torch.float32 on CPU
    """
    model.eval()
    N = x2d_np.shape[0]
    a_list = []

    with torch.no_grad():
        for s in range(0, N, infer_bs):
            e = min(s + infer_bs, N)
            xb = torch.from_numpy(x2d_np[s:e]).to(device, non_blocking=True)  # (bs, bands)
            _, a = model(xb)  # (bs, K)
            a_list.append(a.detach().cpu())
            # 释放显存峰值
            del xb, a

    a_cpu = torch.cat(a_list, dim=0)  # (N, K) CPU
    return a_cpu


def save_abundance_rgb_cpu(a_cpu: torch.Tensor, H: int, W: int, out_path: Path):
    """
    用 CPU 做可视化，进一步节省显存
    Args:
        a_cpu: (H*W, 5) CPU tensor
    """

    colors = COLORS.cpu()  # (5,3)
    rgb = a_cpu @ colors   # (H*W,3)
    rgb = rgb.view(H, W, 3).clamp(0.0, 1.0).numpy()

    plt.imsave(str(out_path), rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True,
                        help="folder containing test mat files")
    parser.add_argument("--mat_key", type=str, default="hyper_image")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="path to trained model (best.pth or latest.pth)")
    parser.add_argument("--out_dir", type=str, default="test_results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--infer_bs", type=int, default=8192,
                        help="inference batch size (pixels per chunk). If OOM, reduce to 4096/2048.")
    parser.add_argument("--endmembers_dir", type=str,
                        default="./endmembers",
                        help="folder containing bk_end.mat, fat_dry_end.mat, fat_fresh_end.mat, lean_dry_end.mat, lean_fresh_end.mat")
    args = parser.parse_args()

    # =========================
    # 单卡 3090
    # =========================
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 加载模型 + 端元
    # =========================
    ckpt = torch.load(args.ckpt, map_location="cpu")
    bands = int(ckpt["bands"])
    K = int(ckpt["K"])

    E = load_endmembers_from_folder(args.endmembers_dir, bands=bands)  # (K,bands) CPU tensor
    if isinstance(E, tuple):
        # 兼容可能返回 (E, class_names) 的版本
        E = E[0]
    if E.shape[0] != K:
        raise ValueError(f"Endmembers K mismatch: ckpt K={K}, loaded E shape={tuple(E.shape)}")

    model = Unmix1DCNNAE(
        bands=bands,
        K=K,
        endmembers=E
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print(f"[INFO] Loaded model from {args.ckpt}")
    print(f"[INFO] bands={bands}, K={K}")
    print(f"[INFO] infer_bs={args.infer_bs}")

    # =========================
    # 测试数据
    # =========================
    mat_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.mat")))
    if len(mat_paths) == 0:
        raise FileNotFoundError(f"No test mat files found in: {args.test_dir}")

    for p in mat_paths:
        name = Path(p).stem
        print(f"[TEST] {name}")

        x2d_np, H, W = load_mat_cube(p, key=args.mat_key)  # numpy (N,bands)
        N = x2d_np.shape[0]
        print(f"  cube -> pixels={N}, H={H}, W={W}, bands={x2d_np.shape[1]}")

        # 按 chunk 推理，拿到 CPU 上的 abundance
        a_cpu = infer_abundance_in_chunks(model, x2d_np, device, infer_bs=args.infer_bs)  # (N,K) CPU

        # 保存彩色丰度图（CPU 可视化，不占显存）
        out_png = out_dir / f"{name}_abundance.png"
        save_abundance_rgb_cpu(a_cpu, H, W, out_png)

        print(f"  saved: {out_png}")

        # 可选：清一下缓存
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()