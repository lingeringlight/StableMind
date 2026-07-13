import torch
import torch.nn as nn



class LearnableNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):  # x: [B, 1, D]
        mu  = x.mean(dim=-1, keepdim=True)  # Bx1x1
        var = x.var (dim=-1, keepdim=True, unbiased=False)  # Bx1x1
        xh = (x - mu) / (var + self.eps).sqrt() # for each channel, learn a new scale
        return self.gamma * xh + self.beta

class ConditionalLearnableNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma_mlp = nn.Linear(dim, dim)
        self.beta_mlp = nn.Linear(dim, dim)

    def forward(self, x):  # x: [B, 1, 4096]
        if x.dim() == 3:
            x = x.flatten(1)
        mu  = x.mean(dim=-1, keepdim=True)
        var = x.var (dim=-1, keepdim=True, unbiased=False)
        xh = (x - mu) / (var + self.eps).sqrt()

        gamma = self.gamma_mlp(x)
        beta = self.beta_mlp(x)
        return gamma * xh + beta


class FourierSBMM(nn.Module):
    def __init__(self, dim, eps=1e-6, hidden=None, mod_phase=False):
        super().__init__()
        self.eps = eps
        h = hidden or dim // 2

        self.gammaA = nn.Sequential(
            nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, 1)
        )
        self.betaA  = nn.Sequential(
            nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, 1)
        )

        if mod_phase:
            self.gammaP = nn.Sequential(
                nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, dim)
            )
            self.betaP  = nn.Sequential(
                nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, dim)
            )
        else:
            self.gammaP = self.betaP = None
        self.mod_phase = mod_phase

        for m in [self.gammaA, self.betaA, self.gammaP, self.betaP]:
            if m is not None:
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)

    def forward(self, x):

        F = torch.fft.rfft(x, dim=-1)
        A, P = torch.abs(F), torch.angle(F)

        muA = A.mean(dim=-1, keepdim=True)
        stdA = A.std(dim=-1, keepdim=True, unbiased=False) + self.eps
        A_norm = (A - muA) / stdA

        muP = P.mean(dim=-1, keepdim=True)
        stdP = P.std(dim=-1, keepdim=True, unbiased=False) + self.eps
        P_norm = (P - muP) / stdP

        gammaA = 1.0 + F.softplus(self.gammaA(x))
        betaA  = self.betaA(x)
        A_mod = gammaA * A_norm + betaA

        if self.mod_phase:
            gammaP = 1.0 + F.softplus(self.gammaP(x))
            betaP  = self.betaP(x)
            P_mod = gammaP * P_norm + betaP
        else:
            P_mod = P

        F_mod = A_mod * torch.exp(1j * P_mod)
        y = torch.fft.irfft(F_mod, n=x.size(-1), dim=-1)
        return y

class SpectralGate(nn.Module):
    def __init__(self, D: int, init_bias: float = 3.0):
        """
        D: V or hidden_dim
        init_bias: Sigmoid(bias)~0.95 -> 初期近似全通，训练逐步学会抑制噪声频率
        """
        super().__init__()
        self.D = D
        self.nfreq = D // 2 + 1
        self.bias = nn.Parameter(torch.ones(self.nfreq) * init_bias)

    def forward(self, x):
        """
        x: [..., D] 或 [B, 1, D] / [B, D] 皆可
        """
        orig_shape = x.shape
        x = x.reshape(-1, self.D)
        with torch.cuda.amp.autocast(enabled=False):  # 关闭混合精度
            x32 = x.to(torch.float32)
            X = torch.fft.rfft(x32, dim=-1)                           # [B*, nfreq]
            gate = torch.sigmoid(self.bias)[None, :]                # [1, nfreq]
            Xg = X * gate
            y = torch.fft.irfft(Xg, n=self.D, dim=-1)               # [B*, D]
            y = y.to(x.dtype)
        return y.reshape(*orig_shape)


class BrainPost(nn.Module):
    def __init__(self, dim, enable_learnnorm=True, init_tau=0.07):
        super().__init__()
        self.enable_learnnorm = enable_learnnorm
        self.ln = LearnableNorm(dim) if enable_learnnorm else nn.Identity()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(init_tau))))

    def forward(self, feats):  # feats: [B, 256, 1664] 或 flatten 后 [B, D]
        if feats.dim() == 3:
            feats = feats.flatten(1)            # [B, 256*1664]
        feats = self.ln(feats) if self.enable_learnnorm else feats
        feats = nn.functional.normalize(feats, dim=-1)
        feats = feats / self.log_tau.exp()      # 温度缩放
        return feats