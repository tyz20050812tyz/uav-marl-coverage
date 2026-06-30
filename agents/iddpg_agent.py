"""IDDPG（Independent DDPG）智能体。

每个智能体独立维护自己的 Actor、Critic 和 Replay Buffer。
Critic 只使用该智能体自身的局部观测和动作。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List
import copy

from .base_agent import BaseAgent
from networks.actor import Actor
from networks.critic import Critic
from utils.replay_buffer import ReplayBufferTensor
from utils.ou_noise import OUNoise


class IDDPGAgent(BaseAgent):
    """独立 DDPG 智能体。"""

    def __init__(self, name: str, obs_dim: int, act_dim: int = 5,
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 gamma: float = 0.95, tau: float = 0.01,
                 buffer_capacity: int = 1000000,
                 device: Optional[torch.device] = None,
                 use_private_buffer: bool = True):
        """
        初始化 IDDPG 智能体。

        Args:
            name: 智能体名称
            obs_dim: 观测维度
            act_dim: 动作维度
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            gamma: 折扣因子
            tau: 软更新系数
            buffer_capacity: 回放池容量
            device: 计算设备
            use_private_buffer: 是否创建私有回放池（IDDPGManager 管理时应设为 False）
        """
        super().__init__(name=name)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Actor 网络 + Target
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)

        # Critic 网络 + Target（IDDPG: 只用自己的 obs + action）
        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 经验回放池（由 use_private_buffer 控制是否创建私有 buffer）
        if use_private_buffer:
            self.buffer = ReplayBufferTensor(capacity=buffer_capacity)
        else:
            self.buffer = None  # 使用外部共享回放池

        # OU 噪声
        self.noise = OUNoise(act_dim)

    def act(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        根据观测选择动作。

        Args:
            obs: (obs_dim,) 观测
            add_noise: 是否添加探索噪声

        Returns:
            (act_dim,) 动作
        """
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()

        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action, 0.0, 1.0)

        return action.astype(np.float32)

    def reset_noise(self):
        """重置 OU 噪声状态。"""
        self.noise.reset()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行一次 DDPG 更新。

        Args:
            batch: 从 replay buffer 采样的 batch，格式:
                {'obs': {name: tensor}, 'actions': {name: tensor},
                 'rewards': {name: tensor}, 'next_obs': {name: tensor},
                 'dones': {name: tensor}}

        Returns:
            {'critic_loss': float, 'actor_loss': float}
        """
        obs = batch['obs'][self.name]
        actions = batch['actions'][self.name]
        rewards = batch['rewards'][self.name]
        next_obs = batch['next_obs'][self.name]
        dones = batch['dones'][self.name]

        # --- 更新 Critic ---
        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            target_q = self.target_critic(next_obs, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(obs, actions)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # --- 更新 Actor ---
        pred_actions = self.actor(obs)
        actor_loss = -self.critic(obs, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 软更新 Target 网络 ---
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """软更新 target 网络参数: θ_target = τ*θ_source + (1-τ)*θ_target。"""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path: str):
        """保存模型。"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt['actor'])
        self.target_actor.load_state_dict(ckpt['target_actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
