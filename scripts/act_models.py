"""act_models.py — ACT（Action Chunking with Transformers）核心机制自实现。

三件套（与官方 ACT 论文一致，代码全部手写）：
  1. action chunking：一次预测未来 k 步动作序列（本实现 k=25）
  2. CVAE：隐变量 z ~ q(z|obs, action_chunk)，推理时从先验 N(0,I) 采样——
     用 z 建模动作的多模态性（同一观测下多种合理动作）
  3. temporal ensemble（推理时）：重叠 chunk 的预测按指数衰减权重
     w = exp(-m * age) 加权平均，m=0.01

架构（state-based 简化版，官方 ACT 的视觉编码器换成 MLP obs 编码器）：
  obs(376) -> MLP -> obs_emb(256)
  CVAE encoder: [obs_emb, action_chunk(25*17)] -> MLP -> mu, logvar (latent 32)
  decoder: 小 Transformer（2 层 4 头 d=256）：tokens = [z_proj, obs_emb+pos_0..24]
           -> 每个位置 token 过线性头 -> 该步动作(17)

边界声明（写进 results/act_vs_bc.md）：
  - 未复现官方 visuomotor 任务（ALOHA 双臂图像操作）——官方 repo 的 Windows 依赖
    （dm_control/labmaze）与双足叙事均不匹配，明确不跑
  - 数据 = SAC 教师示范（RL 教师，非遥操作数据）
  - 形态 = state-based（本体感受），非图像观测
"""
import math

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """观测编码器：obs(376) -> obs_emb(256)。"""

    def __init__(self, obs_dim: int, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class CVAEEncoder(nn.Module):
    """CVAE 后验 q(z | obs_emb, action_chunk) -> mu, logvar。"""

    def __init__(self, emb_dim: int, act_dim: int, chunk: int, latent_dim: int = 32):
        super().__init__()
        in_dim = emb_dim + act_dim * chunk
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)

    def forward(self, obs_emb, action_chunk_flat):
        h = self.net(torch.cat([obs_emb, action_chunk_flat], dim=-1))
        return self.mu_head(h), self.logvar_head(h)


class ChunkDecoder(nn.Module):
    """小 Transformer 解码器：z（style token）+ k 个位置 token -> 每步动作。

    tokens: [z_proj, obs_emb+pos_0, ..., obs_emb+pos_{k-1}]，自注意力让每步
    动作都能看到 z 与全部上下文。z 在官方 ACT 里正是以这种"额外 token"方式注入。
    """

    def __init__(self, emb_dim: int, act_dim: int, chunk: int,
                 latent_dim: int = 32, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.chunk = chunk
        self.z_proj = nn.Linear(latent_dim, emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(chunk, emb_dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=1024,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.action_head = nn.Linear(emb_dim, act_dim)

    def forward(self, obs_emb, z):
        # obs_emb: (B, 256); z: (B, 32)
        z_token = self.z_proj(z).unsqueeze(1)                      # (B, 1, 256)
        obs_tokens = obs_emb.unsqueeze(1) + self.pos_emb.unsqueeze(0)  # (B, k, 256)
        seq = torch.cat([z_token, obs_tokens], dim=1)              # (B, 1+k, 256)
        out = self.transformer(seq)                                # (B, 1+k, 256)
        actions = self.action_head(out[:, 1:])                     # (B, k, act_dim)
        return actions


class ACTModel(nn.Module):
    """ACT 完整模型：obs -> 采样 z（推理）或后验 z（训练）-> 动作 chunk。"""

    def __init__(self, obs_dim: int, act_dim: int, chunk: int = 25,
                 emb_dim: int = 256, latent_dim: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk = chunk
        self.latent_dim = latent_dim
        self.obs_encoder = ObsEncoder(obs_dim, emb_dim)
        self.cvae_encoder = CVAEEncoder(emb_dim, act_dim, chunk, latent_dim)
        self.decoder = ChunkDecoder(emb_dim, act_dim, chunk, latent_dim)

    def encode_obs(self, obs):
        return self.obs_encoder(obs)

    def posterior(self, obs_emb, action_chunk_flat):
        return self.cvae_encoder(obs_emb, action_chunk_flat)

    def decode(self, obs_emb, z):
        return self.decoder(obs_emb, z)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def sample_prior(self, batch_size, device):
        """推理时从先验 N(0, I) 采样 z。"""
        return torch.randn(batch_size, self.latent_dim, device=device)

    def forward(self, obs, action_chunk_flat=None):
        """训练模式（给定动作 chunk 走 CVAE 后验）；推理模式（None 走先验）。"""
        obs_emb = self.encode_obs(obs)
        if action_chunk_flat is not None:
            mu, logvar = self.posterior(obs_emb, action_chunk_flat)
            z = self.reparameterize(mu, logvar)
            pred = self.decode(obs_emb, z)
            return pred, mu, logvar
        z = self.sample_prior(obs.shape[0], obs.device)
        return self.decode(obs_emb, z)


def kl_divergence(mu, logvar) -> torch.Tensor:
    """KL(N(mu, sigma) || N(0, I))，逐样本求和（官方 ACT 的 beta=1 口径）。"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
