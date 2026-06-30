"""评价指标计算模块。

提供覆盖率、碰撞次数、冗余覆盖率、完成步数等指标的计算函数。
"""

import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)


def compute_metrics(agent_positions: np.ndarray,
                    landmark_positions: np.ndarray,
                    coverage_radius: float,
                    collision_threshold: float = 0.1,
                    verbose: bool = False) -> Dict[str, float]:
    """
    计算一组评价指标。

    Args:
        agent_positions: (N, 2) 智能体位置
        landmark_positions: (M, 2) 目标点位置
        coverage_radius: 覆盖半径（距离小于此值视为覆盖）
        collision_threshold: 碰撞距离阈值
        verbose: 是否打印调试信息

    Returns:
        dict with metrics:
            'coverage_rate': 目标覆盖率 [0, 1]
            'collision_count': 碰撞次数
            'avg_min_distance': 平均最小距离
            'redundancy_rate': 冗余覆盖率 [0, 1]
            'covered_landmarks': 被覆盖的目标点数量
    """
    N = agent_positions.shape[0]
    M = landmark_positions.shape[0]

    if N == 0 or M == 0:
        return {
            'coverage_rate': 0.0,
            'collision_count': 0,
            'avg_min_distance': 0.0,
            'redundancy_rate': 0.0,
            'covered_landmarks': 0,
        }

    # 1. 目标覆盖率：被至少一个智能体覆盖的目标点数 / 总目标点数
    covered_landmarks = 0
    landmark_coverage_counts = np.zeros(M, dtype=int)

    for j in range(M):
        distances = np.linalg.norm(agent_positions - landmark_positions[j], axis=1)
        agents_in_range = np.sum(distances <= coverage_radius)
        landmark_coverage_counts[j] = agents_in_range
        if agents_in_range >= 1:
            covered_landmarks += 1

    coverage_rate = covered_landmarks / M

    # 2. 碰撞次数：智能体间距小于碰撞阈值的对数
    collision_count = 0
    for i in range(N):
        for k in range(i + 1, N):
            dist = np.linalg.norm(agent_positions[i] - agent_positions[k])
            if dist < collision_threshold:
                collision_count += 1

    # 3. 平均最小距离：每个目标点到最近智能体距离的均值
    min_distances = []
    for j in range(M):
        dists = np.linalg.norm(agent_positions - landmark_positions[j], axis=1)
        min_distances.append(np.min(dists))
    avg_min_distance = np.mean(min_distances)

    # 4. 冗余覆盖率：有 >= 2 个智能体重复覆盖的目标点数 / 总目标点数
    redundant_landmarks = np.sum(landmark_coverage_counts >= 2)
    redundancy_rate = redundant_landmarks / M

    if verbose:
        logger.info(f"  Coverage: {coverage_rate:.2%} ({covered_landmarks}/{M})")
        logger.info(f"  Collisions: {collision_count}")
        logger.info(f"  Avg Min Dist: {avg_min_distance:.4f}")
        logger.info(f"  Redundancy: {redundancy_rate:.2%} ({redundant_landmarks}/{M})")

    return {
        'coverage_rate': float(coverage_rate),
        'collision_count': int(collision_count),
        'avg_min_distance': float(avg_min_distance),
        'redundancy_rate': float(redundancy_rate),
        'covered_landmarks': int(covered_landmarks),
    }


def compute_completion_steps(agent_positions_history: list,
                             landmark_positions: np.ndarray,
                             coverage_radius: float) -> int:
    """
    计算首次全部目标点被覆盖所需的步数。

    Args:
        agent_positions_history: list of (N, 2) arrays, 每步的智能体位置
        landmark_positions: (M, 2) 目标点位置
        coverage_radius: 覆盖半径

    Returns:
        完成步数，若未完成则返回历史长度
    """
    M = landmark_positions.shape[0]
    for t, positions in enumerate(agent_positions_history):
        covered = set()
        for j in range(M):
            distances = np.linalg.norm(positions - landmark_positions[j], axis=1)
            if np.any(distances <= coverage_radius):
                covered.add(j)
        if len(covered) == M:
            return t + 1
    return len(agent_positions_history)
