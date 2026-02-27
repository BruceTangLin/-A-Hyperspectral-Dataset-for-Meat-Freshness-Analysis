# dataset.py
import os
visible_gpu_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu_id
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # normalize along last dimension
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

@dataclass
class LoadReport:
    num_files: int
    total_pixels: int
    bands: int
    device: str
    dtype: str


def _read_one_mat_to_2d(path: str, key: str = "hyper_image") -> np.ndarray:
    """
    Read one MATLAB v7.3 .mat (HDF5) file using:
        with h5py.File(path,'r') as f:
            cube = np.array(f['hyper_image'])
    cube shape: (bands, samples, lines)
    return X2d shape: (samples*lines, bands) float32
    """
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"[{os.path.basename(path)}] key '{key}' not found. Available keys: {list(f.keys())}")
        cube = np.array(f[key])  # (bands, samples, lines)
    if cube.ndim != 3:
        raise ValueError(f"[{os.path.basename(path)}] expected 3D cube, got {cube.shape}")

    bands, samples, lines = cube.shape
    # (bands, samples, lines) -> (samples, lines, bands) -> (samples*lines, bands)
    x = np.transpose(cube, (1, 2, 0)).reshape(samples * lines, bands)
    return x.astype(np.float32, copy=False)


class HyperMatPreloadDataset(Dataset):
    """
    Preload ALL mat files into a single tensor X of shape (N, bands).
    Optionally move to GPU.
    """
    def __init__(
        self,
        train_dir: str,
        mat_key: str = "hyper_image",
        device: str = "cuda",
        try_load_to_gpu: bool = True,
        pin_memory_if_cpu: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.mat_key = mat_key
        self.verbose = verbose

        mat_paths = sorted(glob.glob(os.path.join(train_dir, "*.mat")))
        if len(mat_paths) == 0:
            raise FileNotFoundError(f"No .mat files found in: {train_dir}")
        self.mat_paths: List[str] = mat_paths

        # ---- load all into CPU first (safer), then try move to GPU ----
        xs: List[np.ndarray] = []
        total = 0
        bands_ref: Optional[int] = None

        if self.verbose:
            print(f"[Dataset] Found {len(mat_paths)} mat files in {train_dir}")

        for p in mat_paths:
            x2d = _read_one_mat_to_2d(p, key=mat_key)  # (n_i, bands)
            if bands_ref is None:
                bands_ref = x2d.shape[1]
            else:
                if x2d.shape[1] != bands_ref:
                    raise ValueError(
                        f"bands mismatch: {os.path.basename(p)} has {x2d.shape[1]}, expected {bands_ref}"
                    )
            xs.append(x2d)
            total += x2d.shape[0]
            if self.verbose:
                print(f"  - {os.path.basename(p)} -> {x2d.shape}, cumulative pixels={total}")

        assert bands_ref is not None
        # concat on CPU
        x_all = np.concatenate(xs, axis=0)  # (N, bands)
        # free list
        xs.clear()

        X_cpu = torch.from_numpy(x_all)  # float32 CPU
        # optional pin memory for faster H2D if we end up training on CPU tensor -> GPU batches
        if pin_memory_if_cpu and X_cpu.device.type == "cpu":
            try:
                X_cpu = X_cpu.pin_memory()
            except Exception:
                pass

        target_device = device
        if try_load_to_gpu and torch.cuda.is_available() and device.startswith("cuda"):
            try:
                X_gpu = X_cpu.to(device, non_blocking=True)
                self.X = X_gpu
                self._device = str(self.X.device)
            except RuntimeError as e:
                # likely CUDA OOM
                print(f"[Dataset] WARNING: move all data to GPU failed: {repr(e)}")
                print("[Dataset] Fallback: keep data on CPU and move per-batch.")
                self.X = X_cpu
                self._device = "cpu"
        else:
            self.X = X_cpu
            self._device = "cpu"

        self.report = LoadReport(
            num_files=len(mat_paths),
            total_pixels=int(self.X.shape[0]),
            bands=int(self.X.shape[1]),
            device=self._device,
            dtype=str(self.X.dtype),
        )

        if self.verbose:
            print(f"[Dataset] Done. X={tuple(self.X.shape)}, dtype={self.X.dtype}, device={self._device}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.X[idx]              # (bands,)
        #x = l2norm(x)                
        return x
    
if __name__ == '__main__':
    ds = HyperMatPreloadDataset(
        train_dir='./train_data',
        mat_key='hyper_image',
        device="0",
        pin_memory_if_cpu=True,
        verbose=True,
    )
    N, bands = ds.report.total_pixels, ds.report.bands
    print(f"[Info] Total pixels: {N}, bands: {bands}, dataset_device: {ds.report.device}")

    x = ds[100000]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(x, linewidth=1.5)
    plt.xlabel("Band Index")
    plt.ylabel("Reflectance")
    plt.grid(True)

    plt.savefig("spectrum.png", dpi=300)
    plt.close()

    print("Saved spectrum.png")