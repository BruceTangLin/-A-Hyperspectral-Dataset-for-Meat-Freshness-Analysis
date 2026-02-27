import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Model: 1D-CNN Encoder -> abundances (K=5)
# Decoder: fixed endmembers E (K x bands), reconstruct x_hat = a @ E
# ---------------------------

class Unmix1DCNNAE(nn.Module):
    def __init__(self, bands: int, K: int, endmembers: torch.Tensor):
        """
        endmembers: shape (K, bands) in float32/float64; will be registered as buffer (fixed)
        """
        super().__init__()
        assert endmembers.shape == (K, bands), f"endmembers must be (K,bands)=({K},{bands}), got {tuple(endmembers.shape)}"
        self.bands = bands
        self.K = K

        # Register fixed endmembers (no grad)
        self.register_buffer("E", endmembers.clone().detach())

        # 1D CNN encoder: input (B,1,bands) -> features -> (B,K)
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # bands/2

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # bands/4

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # global pooling + linear to K
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B,128,1)
            nn.Flatten(),             # (B,128)
            nn.Linear(128, K),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, bands)
        return a: (B, K), non-negative and sum-to-one via softmax
        """
        h = x.unsqueeze(1)         # (B,1,bands)
        h = self.enc(h)            # (B,128,~)
        logits = self.head(h)      # (B,K)
        a = F.softmax(logits, dim=-1)
        return a

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        """
        a: (B,K)
        return x_hat: (B,bands)
        """
        # (B,K) @ (K,bands) -> (B,bands)
        return a @ self.E

    def forward(self, x: torch.Tensor):
        a = self.encode(x)
        x_hat = self.decode(a)
        return x_hat, a
    

if __name__ == '__main__':
    x = torch.randn(25*25,431)
    e = torch.randn(5,431)
    model = Unmix1DCNNAE(431,5,e)
    x_hat, a = model(x)
    print(x_hat.shape)