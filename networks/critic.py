"""Critic 网络模块。

集中式 Critic：接收所有智能体的观测和动作，输出联合 Q 值。
支持截断双 Q 网络（TD3 风格）。
"""

import torch
import torch.nn as nn
from typing import List


class Critic(nn.Module):
    """集中式 Critic 网络：所有智能体的观测+动作 → Q 值。"""

    def __init__(self, all_obs_dim: int, all_act_dim: int,
                 hidden_dims: List[int] = None):
        """
        初始化 Critic 网络。

        Args:
            all_obs_dim: 所有智能体观测的总维度
            all_act_dim: 所有智能体动作的总维度
            hidden_dims: 隐藏层维度列表，默认 [128, 64]
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        input_dim = all_obs_dim + all_act_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, all_obs: torch.Tensor,
                all_actions: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            all_obs: (batch_size, all_obs_dim) 所有智能体的观测拼接
            all_actions: (batch_size, all_act_dim) 所有智能体的动作拼接

        Returns:
            (batch_size, 1) Q 值
        """
        x = torch.cat([all_obs, all_actions], dim=1)
        return self.net(x)


class TwinCritic(nn.Module):
    """
    截断双 Q 网络（TD3 风格）。

    维护两个独立的 Critic，目标 Q 值取二者最小值以压制 Q 值高估。
    """

    def __init__(self, all_obs_dim: int, all_act_dim: int,
                 hidden_dims: List[int] = None):
        """
        初始化双 Critic。

        Args:
            all_obs_dim: 所有智能体观测的总维度
            all_act_dim: 所有智能体动作的总维度
            hidden_dims: 隐藏层维度列表，默认 [128, 64]
        """
        super().__init__()
        self.critic_A = Critic(all_obs_dim, all_act_dim, hidden_dims)
        self.critic_B = Critic(all_obs_dim, all_act_dim, hidden_dims)

    def forward(self, all_obs: torch.Tensor,
                all_actions: torch.Tensor) -> torch.Tensor:
        """
        返回 Q_A 值（默认前向用 critic_A）。
        """
        return self.critic_A(all_obs, all_actions)

    def q_values(self, all_obs: torch.Tensor,
                 all_actions: torch.Tensor) -> tuple:
        """返回 (Q_A, Q_B)。"""
        return (self.critic_A(all_obs, all_actions),
                self.critic_B(all_obs, all_actions))

    def q_min(self, all_obs: torch.Tensor,
              all_actions: torch.Tensor) -> torch.Tensor:
        """返回 min(Q_A, Q_B)，用于 TD target 计算。"""
        qa, qb = self.q_values(all_obs, all_actions)
        return torch.min(qa, qb)
