"""随机策略智能体。

完全不学习，从动作空间均匀随机采样。作为最基础的基线对照组。
"""

import numpy as np
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """随机动作智能体。"""

    def __init__(self, act_dim: int = 5, name: str = "Random"):
        super().__init__(name=name)
        self.act_dim = act_dim

    def act(self, obs, **kwargs):
        """
        从均匀分布采样动作。

        Args:
            obs: 观测（不使用）
        Returns:
            (act_dim,) numpy 数组，值 ∈ [0, 1]
        """
        return np.random.uniform(0, 1, size=self.act_dim).astype(np.float32)

    def update(self, *args, **kwargs):
        """Random agent 不学习。"""
        pass

    def reset(self):
        """重置 agent 状态（无操作）。"""
        pass
