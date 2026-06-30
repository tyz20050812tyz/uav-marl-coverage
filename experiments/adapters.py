"""智能体适配器模块。

将 IDDPG 和 Random 智能体包装为 Trainer 兼容的统一接口。
Trainer 期望 agent 具有:
- act(name, obs, add_noise) → (act_dim,) numpy
- buffer: ReplayBufferTensor (共享回放池)
- update(batch) → {'critic_loss': float, 'actor_loss': float}
- save(path) / load(path)
- reset_noise() (可选)
"""

import numpy as np
import torch
from typing import Dict, Optional

from agents.iddpg_agent import IDDPGAgent
from agents.random_agent import RandomAgent
from utils.replay_buffer import ReplayBufferTensor


class IDDPGManager:
    """管理 N 个独立 IDDPG 智能体，对外暴露 MADDPG 风格接口。"""

    def __init__(self, num_agents: int, obs_dim: int, act_dim: int = 5,
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 gamma: float = 0.95, tau: float = 0.01,
                 buffer_capacity: int = 1000000,
                 device: Optional[torch.device] = None):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.agent_names = [f'agent_{i}' for i in range(num_agents)]
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # 创建 N 个独立 IDDPG 智能体
        self.agents: Dict[str, IDDPGAgent] = {}
        for name in self.agent_names:
            self.agents[name] = IDDPGAgent(
                name=name, obs_dim=obs_dim, act_dim=act_dim,
                actor_lr=actor_lr, critic_lr=critic_lr,
                gamma=gamma, tau=tau,
                buffer_capacity=buffer_capacity,
                device=self.device,
                use_private_buffer=False,  # 使用共享回放池，避免内存浪费
            )

        # 共享回放池（Trainer 兼容接口）
        self.buffer = ReplayBufferTensor(capacity=buffer_capacity)

    def act(self, agent_name: str, obs: np.ndarray,
            add_noise: bool = True) -> np.ndarray:
        """单个智能体选择动作。"""
        return self.agents[agent_name].act(obs, add_noise)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新所有独立智能体。"""
        total_critic = 0.0
        total_actor = 0.0
        for name in self.agent_names:
            info = self.agents[name].update(batch)
            total_critic += info['critic_loss']
            total_actor += info['actor_loss']
        n = len(self.agent_names)
        return {'critic_loss': total_critic / n, 'actor_loss': total_actor / n}

    def reset_noise(self):
        """重置所有智能体的 OU 噪声。"""
        for agent in self.agents.values():
            agent.reset_noise()

    def save(self, path: str):
        """保存所有智能体。"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {}
        for name in self.agent_names:
            a = self.agents[name]
            ckpt[name] = {
                'actor': a.actor.state_dict(),
                'target_actor': a.target_actor.state_dict(),
                'critic': a.critic.state_dict(),
                'target_critic': a.target_critic.state_dict(),
            }
        torch.save(ckpt, path)

    def load(self, path: str):
        """加载所有智能体。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        for name in self.agent_names:
            a = self.agents[name]
            a.actor.load_state_dict(ckpt[name]['actor'])
            a.target_actor.load_state_dict(ckpt[name]['target_actor'])
            a.critic.load_state_dict(ckpt[name]['critic'])
            a.target_critic.load_state_dict(ckpt[name]['target_critic'])


class RandomManager:
    """管理 N 个随机智能体，对外暴露统一接口。

    注意：不设置 buffer 属性，使 Trainer 跳过训练更新。
    """

    def __init__(self, num_agents: int, act_dim: int = 5):
        self.num_agents = num_agents
        self.act_dim = act_dim
        self.agent_names = [f'agent_{i}' for i in range(num_agents)]
        self.agents = {name: RandomAgent(act_dim)
                       for name in self.agent_names}
        self.device = torch.device('cpu')

        # 有意不设置 self.buffer，让 Trainer 的 hasattr 返回 False

    def act(self, agent_name: str, obs: np.ndarray,
            add_noise: bool = True) -> np.ndarray:
        """随机采样动作。"""
        return self.agents[agent_name].act(obs)

    def update(self, batch=None) -> Dict[str, float]:
        """Random agent 不学习。"""
        return {'critic_loss': 0.0, 'actor_loss': 0.0}

    def reset_noise(self):
        """无需重置。"""
        pass

    def save(self, path: str):
        """无需保存。"""
        pass

    def load(self, path: str):
        """无需加载。"""
        pass
