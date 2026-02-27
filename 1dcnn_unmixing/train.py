# train.py
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from dataset import HyperMatPreloadDataset
from model import Unmix1DCNNAE  # uses fixed endmembers buffer E :contentReference[oaicite:1]{index=1}
from load_endmember import load_endmembers_from_folder


def unmix_loss(x: torch.Tensor, x_hat: torch.Tensor, a: torch.Tensor,
              w_sum: float = 0.01, w_sparse: float = 0.001):
    mse = F.mse_loss(x_hat, x)

    # # sum-to-one (softmax already enforces, but keep as requested)
    # lsum = ((a.sum(dim=-1) - 1.0) ** 2).mean()

    # # sparsity (L1)
    # lsparse = a.abs().mean()

    loss = mse  
    return loss, mse.detach()  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="train_mat")
    parser.add_argument("--mat_key", type=str, default="hyper_image")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use on multi-GPU server")
    parser.add_argument("--preload_gpu", action="store_true", help="try to move ALL data to GPU (may OOM)")
    parser.add_argument("--endmembers_pth", type=str, default="endmembers")
    parser.add_argument("--out_dir", type=str, default="runs/unmix_1dcnn")
    # parser.add_argument("--w_sum", type=float, default=0.01)
    # parser.add_argument("--w_sparse", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- select one GPU on multi-GPU server ----
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- dataset (preload all) ----
    ds = HyperMatPreloadDataset(
        train_dir=args.train_dir,
        mat_key=args.mat_key,
        device=str(device),
        try_load_to_gpu=bool(args.preload_gpu),
        pin_memory_if_cpu=True,
        verbose=True,
    )
    N, bands = ds.report.total_pixels, ds.report.bands
    print(f"[Info] Total pixels: {N}, bands: {bands}, dataset_device: {ds.report.device}")

    # ---- load endmembers and build model ----
    E = load_endmembers_from_folder(args.endmembers_pth, bands=bands)  # CPU tensor
    model = Unmix1DCNNAE(bands=bands, K=args.K, endmembers=E).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_epoch_loss = float("inf")
    best_path = out_dir / "best.pth"
    latest_path = out_dir / "latest.pth"

    # For speed when dataset is on GPU: do manual batching by indexing
    X = ds.X  # could be CPU or GPU tensor
    dataset_on_gpu = (X.device.type == "cuda")
    print(f"[Info] Training device: {device}, X device: {X.device}, preload_gpu={args.preload_gpu}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        # shuffle indices (put indices on same device as X for fast indexing if X is CUDA)
        idx_device = X.device
        perm = torch.randperm(N, device=idx_device)

        running = 0.0
        running_mse = 0.0
        # running_lsum = 0.0
        # running_lsparse = 0.0
        steps = 0

        pbar = tqdm(range(0, N, args.batch_size), desc=f"Epoch {epoch:03d}/{args.epochs}", ncols=120)
        for start in pbar:
            end = min(start + args.batch_size, N)
            batch_idx = perm[start:end]

            # get batch
            xb = X.index_select(0, batch_idx)
            # if X on CPU, move batch to GPU
            if not dataset_on_gpu and device.type == "cuda":
                xb = xb.to(device, non_blocking=True)
            elif device.type == "cpu":
                xb = xb.to("cpu")

            optim.zero_grad(set_to_none=True)
            x_hat, a = model(xb)
            #loss, mse, lsum, lsparse = unmix_loss(xb, x_hat, a, w_sum=args.w_sum, w_sparse=args.w_sparse)
            loss, mse = unmix_loss(xb, x_hat, a)
            loss.backward()
            optim.step()

            running += float(loss.detach().item())
            running_mse += float(mse.item())
            # running_lsum += float(lsum.item())
            # running_lsparse += float(lsparse.item())
            steps += 1

            pbar.set_postfix({
                "loss": f"{running/steps:.6f}",
                "mse": f"{running_mse/steps:.6f}",
            })

        epoch_loss = running / max(steps, 1)

        # ---- save latest ----
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "best_epoch_loss": best_epoch_loss,
            "epoch_loss": epoch_loss,
            "bands": bands,
            "K": args.K,
        }, str(latest_path))

        # ---- save best ----
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "best_epoch_loss": best_epoch_loss,
                "epoch_loss": epoch_loss,
                "bands": bands,
                "K": args.K,
            }, str(best_path))

        print(f"[Epoch {epoch:03d}] epoch_loss={epoch_loss:.6f} | best={best_epoch_loss:.6f} | saved: latest.pth, best.pth")


if __name__ == "__main__":
    main()