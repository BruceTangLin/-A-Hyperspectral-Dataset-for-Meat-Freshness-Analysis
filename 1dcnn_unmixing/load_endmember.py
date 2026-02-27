import os
import numpy as np
import h5py
import torch

def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)

def _read_mat_vector_v73(path: str, key_candidates=None) -> np.ndarray:
    """
    Read one v7.3 .mat (HDF5) file and return a 1D float32 vector of shape (bands,).
    """
    if key_candidates is None:
        key_candidates = ["spec_avg", "spec", "spectrum", "endmember", "E", "x"]

    with h5py.File(path, "r") as f:
        found = None
        for k in key_candidates:
            if k in f:
                found = k
                break
        if found is None:
            raise KeyError(
                f"[{os.path.basename(path)}] cannot find key in {key_candidates}. "
                f"Available keys: {list(f.keys())}"
            )
        arr = np.array(f[found])

    # squeeze to 1D
    arr = np.array(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"[{os.path.basename(path)}] expected 1D vector after squeeze, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_endmembers_from_folder(endmembers_dir: str, bands: int) -> torch.Tensor:
    """
    Build endmember matrix E with shape (K, bands) from 5 .mat files.
    Order (K=5):
        0: bk_end.mat
        1: fat_dry_end.mat
        2: fat_fresh_end.mat
        3: lean_dry_end.mat
        4: lean_fresh_end.mat
    Each file contains a vector (bands,).
    """
    names = [
        "bk_end.mat",
        "fat_dry_end.mat",
        "fat_fresh_end.mat",
        "lean_dry_end.mat",
        "lean_fresh_end.mat",
    ]
    vecs = []
    for n in names:
        p = os.path.join(endmembers_dir, n)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Endmember file not found: {p}")

        v = _read_mat_vector_v73(p)  # (bands,)
        if v.shape[0] != bands:
            raise ValueError(
                f"[{n}] bands mismatch: got {v.shape[0]}, expected {bands}. "
                f"(Maybe your cube bands order or stored vector length differs.)"
            )
        vecs.append(v)

    E = np.stack(vecs, axis=0)  # (K, bands)
    E = torch.from_numpy(E).float()
    #E = l2norm(E)
    return E

if __name__ == '__main__':
    pth = './endmembers'
    bands = 431
    e = load_endmembers_from_folder(pth,bands)
    print(e.shape)

    import matplotlib.pyplot as plt
    spectrum = e[1, :] 

    plt.figure(figsize=(8,5))
    plt.plot(spectrum, linewidth=1.5)
    plt.xlabel("Band Index")
    plt.ylabel("Reflectance")
    plt.grid(True)

    plt.savefig("spectrum_200_400.png", dpi=300)
    plt.close()

    print("Saved spectrum_200_400.png")