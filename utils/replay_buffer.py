"""经验回放池模块。

支持多智能体的集中式经验回放，每个 transition 包含：
- obs: 所有智能体的观测 (dict)
- actions: 所有智能体的动作 (dict)
- rewards: 所有智能体的奖励 (dict)
- next_obs: 下一时刻所有智能体的观测 (dict)
- dones: 终止标记 (dict)
"""

from collections import deque
import random
import numpy as np
from typing import Dict, List, Any, Optional
import torch


class ReplayBuffer:
    """固定容量的经验回放池。"""

    def __init__(self, capacity: int = 1000000):
        """
        初始化回放池。

        Args:
            capacity: 最大存储的 transition 数量
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs: Dict[str, np.ndarray],
             actions: Dict[str, np.ndarray],
             rewards: Dict[str, float],
             next_obs: Dict[str, np.ndarray],
             dones: Dict[str, bool]):
        """存入一条 transition。"""
        self.buffer.append({
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
        })

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        随机采样一个 batch。

        Args:
            batch_size: 批次大小

        Returns:
            dict with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones'
            每个值都是对应的 numpy 数组或 list
        """
        batch_size = min(batch_size, len(self.buffer))
        samples = random.sample(list(self.buffer), batch_size)

        obs_batch = [s['obs'] for s in samples]
        actions_batch = [s['actions'] for s in samples]
        rewards_batch = [s['rewards'] for s in samples]
        next_obs_batch = [s['next_obs'] for s in samples]
        dones_batch = [s['dones'] for s in samples]

        return {
            'obs': obs_batch,
            'actions': actions_batch,
            'rewards': rewards_batch,
            'next_obs': next_obs_batch,
            'dones': dones_batch,
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """检查是否积累了足够数据来采样一个 batch。"""
        return len(self.buffer) >= batch_size


class ReplayBufferTensor:
    """Tensor 版本的经验回放池，直接返回 torch tensor。"""

    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs: Dict[str, np.ndarray],
             actions: Dict[str, np.ndarray],
             rewards: Dict[str, float],
             next_obs: Dict[str, np.ndarray],
             dones: Dict[str, bool]):
        """存入一条 transition。"""
        self.buffer.append({
            'obs': {k: v.astype(np.float32) for k, v in obs.items()},
            'actions': {k: v.astype(np.float32) for k, v in actions.items()},
            'rewards': {k: np.float32(v) for k, v in rewards.items()},
            'next_obs': {k: v.astype(np.float32) for k, v in next_obs.items()},
            'dones': {k: np.float32(v) for k, v in dones.items()},
        })

    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        随机采样一个 batch，返回 torch tensor。

        Args:
            batch_size: 批次大小
            device: torch device

        Returns:
            dict with tensors
        """
        batch_size = min(batch_size, len(self.buffer))
        samples = random.sample(list(self.buffer), batch_size)

        # 获取 agent names（假设所有样本一致）
        agent_names = list(samples[0]['obs'].keys())

        obs = {}
        actions = {}
        rewards = {}
        next_obs = {}
        dones = {}

        for name in agent_names:
            obs[name] = torch.tensor(
                [s['obs'][name] for s in samples], dtype=torch.float32
            ).to(device)
            actions[name] = torch.tensor(
                [s['actions'][name] for s in samples], dtype=torch.float32
            ).to(device)
            rewards[name] = torch.tensor(
                [s['rewards'][name] for s in samples], dtype=torch.float32
            ).unsqueeze(1).to(device)
            next_obs[name] = torch.tensor(
                [s['next_obs'][name] for s in samples], dtype=torch.float32
            ).to(device)
            dones[name] = torch.tensor(
                [s['dones'][name] for s in samples], dtype=torch.float32
            ).unsqueeze(1).to(device)

        return {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size
