"""安全距离约束模块。

在无人机间距离小于安全距离阈值但尚未碰撞时，给予渐进惩罚，
使智能体提前学会保持安全距离。
"""

import numpy as np


class SafetyPenalty:
    """安全距离约束惩罚计算器。"""

    def __init__(self, safe_distance: float, lambda_s_max: float = 0.5):
        """
        初始化安全距离约束模块。

        Args:
            safe_distance: 安全距离阈值（距离 < 此值时触发惩罚）
            lambda_s_max: 最大安全距离惩罚权重
        """
        self.safe_distance = safe_distance
        self.lambda_s_max = lambda_s_max
        self._current_lambda = lambda_s_max

    def set_lambda(self, value: float):
        """动态调整惩罚权重（供 weight_scheduler 调用）。"""
        self._current_lambda = value

    def compute(self, agent_positions: np.ndarray) -> tuple:
        """
        计算安全距离惩罚。

        Args:
            agent_positions: (N, 2) 智能体位置

        Returns:
            (reward_dict, safety_info)
            reward_dict: {agent_name: penalty_value}
            safety_info: {'violations': int, 'total_penalty': float}
        """
        N = agent_positions.shape[0]
        violations = 0
        total_penalty = 0.0

        # 计算所有智能体对的间距
        for i in range(N):
            for k in range(i + 1, N):
                dist = np.linalg.norm(agent_positions[i] - agent_positions[k])
                if dist < self.safe_distance:
                    # 渐进惩罚：距离越近惩罚越大
                    penalty = self._current_lambda * (self.safe_distance - dist)
                    total_penalty -= penalty
                    violations += 1

        # 将惩罚均匀分配
        rewards = {}
        penalty_per_agent = total_penalty / max(N, 1)
        for i in range(N):
            rewards[f'agent_{i}'] = penalty_per_agent

        return rewards, {
            'violations': violations,
            'total_penalty': total_penalty,
        }
