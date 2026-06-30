"""Ornstein-Uhlenbeck 探索噪声模块。

OU 过程产生时间相关的噪声，适合惯性系统（如无人机）的探索策略。
"""

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck 过程噪声生成器。"""

    def __init__(self, action_dim: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.2,
                 seed: int = None):
        """
        初始化 OU 噪声。

        Args:
            action_dim: 动作空间维度
            mu: 均值（长期均值）
            theta: 均值回归速率（越大回归越快）
            sigma: 波动幅度
            seed: 随机种子
        """
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self._state = None
        if seed is not None:
            np.random.seed(seed)
        self.reset()

    def reset(self):
        """重置噪声状态到均值。"""
        self._state = self.mu.copy()

    def sample(self) -> np.ndarray:
        """
        采样一步噪声。

        Returns:
            (action_dim,) 噪声向量
        """
        dx = self.theta * (self.mu - self._state) + \
             self.sigma * np.random.randn(self.action_dim)
        self._state = self._state + dx
        return self._state.copy()

    def sample_clipped(self, clip_min: float = 0.0,
                       clip_max: float = 1.0) -> np.ndarray:
        """采样并截断到指定范围。"""
        noise = self.sample()
        return np.clip(noise, clip_min, clip_max)
