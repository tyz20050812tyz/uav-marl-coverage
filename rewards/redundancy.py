"""冗余覆盖惩罚模块。

当同一目标点的覆盖半径内存在超过 1 个智能体时触发惩罚，
惩罚与超出智能体数量成正比。
"""

import numpy as np


class RedundancyPenalty:
    """冗余覆盖惩罚计算器。"""

    def __init__(self, coverage_radius: float, lambda_r_max: float = 0.3):
        """
        初始化冗余惩罚模块。

        Args:
            coverage_radius: 覆盖半径（距离 ≤ 此值视为覆盖）
            lambda_r_max: 最大冗余惩罚权重
        """
        self.coverage_radius = coverage_radius
        self.lambda_r_max = lambda_r_max
        self._current_lambda = lambda_r_max  # 默认使用最大权重，可由 scheduler 调整

    def set_lambda(self, value: float):
        """动态调整惩罚权重（供 weight_scheduler 调用）。"""
        self._current_lambda = value

    def compute(self, agent_positions: np.ndarray,
                landmark_positions: np.ndarray) -> tuple:
        """
        计算冗余覆盖惩罚。

        Args:
            agent_positions: (N, 2) 智能体位置
            landmark_positions: (M, 2) 目标点位置

        Returns:
            (reward_dict, coverage_info)
            reward_dict: {agent_name: penalty_value}
            coverage_info: {'coverage_counts': (M,) array, 'redundant_count': int}
        """
        N = agent_positions.shape[0]
        M = landmark_positions.shape[0]

        # 计算每个目标点覆盖半径内的智能体数量
        coverage_counts = np.zeros(M, dtype=int)
        for j in range(M):
            distances = np.linalg.norm(
                agent_positions - landmark_positions[j], axis=1
            )
            coverage_counts[j] = np.sum(distances <= self.coverage_radius)

        # 冗余惩罚：R = -lambda * sum_j max(0, c_j - 1)
        redundant_count = np.sum(np.maximum(coverage_counts - 1, 0))
        total_penalty = -self._current_lambda * redundant_count

        # 将惩罚均匀分配给所有在覆盖区域内的智能体
        rewards = {}
        penalty_per_agent = total_penalty / max(N, 1)
        for i in range(N):
            rewards[f'agent_{i}'] = penalty_per_agent

        return rewards, {
            'coverage_counts': coverage_counts,
            'redundant_count': int(redundant_count),
            'total_penalty': total_penalty,
        }
