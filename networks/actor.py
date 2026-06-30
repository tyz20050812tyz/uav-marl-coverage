"""Actor 网络模块。

Actor 输出 5 维连续动作，对应 [不动, 左移, 右移, 下移, 上移] 五个方向的动作强度。
输出层使用 Sigmoid 将动作值约束到 [0, 1] 区间。
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np


class Actor(nn.Module):
    """Actor 网络：观测 → 连续动作。"""

    def __init__(self, obs_dim: int, act_dim: int = 5,
                 hidden_dims: list = None):
        """
        初始化 Actor 网络。

        Args:
            obs_dim: 观测维度
            act_dim: 动作维度（默认 5）
            hidden_dims: 隐藏层维度列表，默认 [64, 64]
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        layers = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, act_dim))
        layers.append(nn.Sigmoid())  # 将输出约束到 [0, 1]

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            obs: (batch_size, obs_dim) 观测张量

        Returns:
            (batch_size, act_dim) 动作张量，值 ∈ [0, 1]
        """
        return self.net(obs)

    def get_action(self, obs: torch.Tensor,
                   noise: Optional[float] = None,
                   noise_clip: float = 0.5) -> np.ndarray:
        """
        获取带探索噪声的动作（用于环境交互）。

        Args:
            obs: (obs_dim,) 单个观测
            noise: 噪声值（如 OU 噪声样本），若为 None 则不添加噪声
            noise_clip: 噪声截断范围 [-noise_clip, +noise_clip]

        Returns:
            (act_dim,) 动作 numpy 数组，值 ∈ [0, 1]
        """
        self.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            action = self.forward(obs).squeeze(0).cpu().numpy()

        if noise is not None:
            action = action + noise
            action = np.clip(action, 0.0, 1.0)

        return action

    def get_deterministic_action(self, obs: torch.Tensor) -> np.ndarray:
        """获取确定性动作（无噪声，用于评估）。"""
        return self.get_action(obs, noise=None)
