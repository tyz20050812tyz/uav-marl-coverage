"""动态奖励权重调度器。

分阶段调整各奖励项权重：
- 早期：偏重覆盖，轻微惩罚
- 中期：逐步增加协作惩罚
- 后期：全面启用所有约束

支持从环境初始状态动态标定权重，使不同奖励项量级对齐。
"""

import numpy as np
from typing import Dict, Optional


class WeightScheduler:
    """分段线性权重调度器。

    支持从环境初始状态（智能体与目标点平均距离）动态标定初始权重，
    使 Assignment、Redundancy、Safety、Collision 四个奖励项的量级对齐。
    """

    def __init__(self, total_episodes: int,
                 early_ratio: float = 0.3,
                 middle_ratio: float = 0.4,
                 calibrated_lambda_s_max: Optional[float] = None,
                 calibrated_lambda_r_max: Optional[float] = None,
                 calibrated_lambda_a: Optional[float] = None,
                 calibrated_lambda_c_max: Optional[float] = None):
        """
        初始化权重调度器。

        Args:
            total_episodes: 总训练 episode 数
            early_ratio: 早期阶段占总 episode 的比例（0-30%）
            middle_ratio: 中期阶段占比（30-70%）
            后期自动计算为 1 - early - middle
            calibrated_lambda_s_max: 动态标定后的安全惩罚最大权重（None 则使用调用时传入的原始值）
            calibrated_lambda_r_max: 动态标定后的冗余惩罚最大权重
            calibrated_lambda_a: 动态标定后的分配奖励权重
            calibrated_lambda_c_max: 动态标定后的碰撞惩罚最大权重
        """
        self.total_episodes = total_episodes
        self.early_episodes = int(total_episodes * early_ratio)
        self.middle_episodes = int(total_episodes * middle_ratio)
        self.late_start = self.early_episodes + self.middle_episodes

        # 动态标定后的权重（None 表示未标定，使用原始值）
        self._calibrated = {
            'lambda_s_max': calibrated_lambda_s_max,
            'lambda_r_max': calibrated_lambda_r_max,
            'lambda_a': calibrated_lambda_a,
            'lambda_c_max': calibrated_lambda_c_max,
        }

    def get_weights(self, episode: int,
                    lambda_s_max: float,
                    lambda_r_max: float,
                    lambda_a: float,
                    lambda_c_max: float = 0.5) -> Dict[str, float]:
        """
        获取当前 episode 的奖励权重。

        优先使用动态标定值（如果已标定），否则使用传入的原始值。

        Args:
            episode: 当前 episode 编号（0-indexed）
            lambda_s_max: 安全距离惩罚最大权重（原始值，标定后自动替换）
            lambda_r_max: 冗余惩罚最大权重（原始值，标定后自动替换）
            lambda_a: 分配奖励权重（原始值，标定后自动替换）
            lambda_c_max: 碰撞惩罚最大权重（原始值，标定后自动替换）

        Returns:
            {'lambda_a': float, 'lambda_s': float, 'lambda_r': float, 'lambda_c': float}
        """
        # 使用动态标定值（如果可用）
        s_max = self._calibrated['lambda_s_max'] if self._calibrated['lambda_s_max'] is not None else lambda_s_max
        r_max = self._calibrated['lambda_r_max'] if self._calibrated['lambda_r_max'] is not None else lambda_r_max
        a_val = self._calibrated['lambda_a'] if self._calibrated['lambda_a'] is not None else lambda_a
        c_max = self._calibrated['lambda_c_max'] if self._calibrated['lambda_c_max'] is not None else lambda_c_max
        if episode < self.early_episodes:
            # 早期：恒定 0.2 * s_max, 0.1 * r_max, 0.3 * c_max（初始即启用基础惩罚）
            lambda_s = 0.2 * s_max
            lambda_r = 0.1 * r_max
            lambda_c = 0.3 * c_max
        elif episode < self.late_start:
            # 中期：0.2→0.5 * s_max, 0.1→0.5 * r_max, 0.3→0.6 * c_max
            t = (episode - self.early_episodes) / max(self.middle_episodes, 1)
            lambda_s = s_max * (0.2 + 0.3 * t)
            lambda_r = r_max * (0.1 + 0.4 * t)
            lambda_c = c_max * (0.3 + 0.3 * t)
        else:
            # 后期：0.5→1.0 * s_max, 0.5→1.0 * r_max, 0.6→1.0 * c_max
            late_episodes = self.total_episodes - self.late_start
            t = min((episode - self.late_start) / max(late_episodes, 1), 1.0)
            lambda_s = s_max * (0.5 + 0.5 * t)
            lambda_r = r_max * (0.5 + 0.5 * t)
            lambda_c = c_max * (0.6 + 0.4 * t)

        return {
            'lambda_a': a_val,
            'lambda_s': lambda_s,
            'lambda_r': lambda_r,
            'lambda_c': lambda_c,
        }

    def get_stage(self, episode: int) -> str:
        """返回当前训练阶段名称。"""
        if episode < self.early_episodes:
            return 'early'
        elif episode < self.late_start:
            return 'middle'
        else:
            return 'late'

    @staticmethod
    def calibrate_from_state(agent_positions: np.ndarray,
                             landmark_positions: np.ndarray,
                             lambda_s_max: float = 0.5,
                             lambda_r_max: float = 0.3,
                             lambda_a: float = 0.5,
                             lambda_c_max: float = 0.5,
                             safe_distance: float = 0.2,
                             coverage_radius: float = 0.24,
                             collision_threshold: float = 0.15,
                             target_magnitude: float = 1.0) -> Dict[str, float]:
        """从环境初始状态动态标定权重，使各奖励项量级对齐。

        核心思路：
        1. 计算初始状态下的典型距离尺度（agent-landmark 平均距离 d_al、agent-agent 平均距离 d_aa）
        2. 推算各奖励项在当前距离下的典型值
        3. 缩放 lambda 使各奖励项初始量级接近 target_magnitude

        Args:
            agent_positions: (N, 2) 初始智能体位置
            landmark_positions: (M, 2) 初始目标点位置
            lambda_s_max: 原始安全惩罚最大权重
            lambda_r_max: 原始冗余惩罚最大权重
            lambda_a: 原始分配奖励权重
            lambda_c_max: 原始碰撞惩罚最大权重
            safe_distance: 安全距离阈值
            coverage_radius: 覆盖半径
            collision_threshold: 碰撞距离阈值
            target_magnitude: 目标奖励量级（各奖励项初始值缩放到此附近）

        Returns:
            {'lambda_s_max': float, 'lambda_r_max': float, 'lambda_a': float, 'lambda_c_max': float}
            标定后的权重，可直接传入 WeightScheduler 构造函数
        """
        N = agent_positions.shape[0]
        M = landmark_positions.shape[0]

        # 1. 计算初始距离尺度
        # agent-landmark 平均最小距离
        al_dists = []
        for j in range(M):
            d = np.linalg.norm(agent_positions - landmark_positions[j], axis=1)
            al_dists.append(np.min(d))
        d_al = np.mean(al_dists) if al_dists else 1.0

        # agent-agent 平均距离
        aa_dists = []
        for i in range(N):
            for k in range(i + 1, N):
                aa_dists.append(np.linalg.norm(agent_positions[i] - agent_positions[k]))
        d_aa = np.mean(aa_dists) if aa_dists else 1.0

        logger_ws = __import__('logging').getLogger(__name__)
        logger_ws.info(
            "[WeightScheduler] 初始距离尺度: d_al=%.3f, d_aa=%.3f", d_al, d_aa
        )

        # 2. 推算各奖励项在初始状态下的典型值（未缩放）
        #    Assignment: R_assign ≈ -λa * d_al （每个 agent 到最近 landmark 的距离）
        #    Safety: 若 d_aa > safe_distance 则不触发；否则典型值 ≈ max(0, dsafe - d_aa)
        #    Collision: 若 d_aa > collision_threshold 则不触发；否则 ≈ -(threshold - d_aa)/threshold
        #    Redundancy: 初始通常不触发（各 agent 分散），使用启发式估计

        typical_assign = d_al  # 无 λ 时的典型值
        typical_safety = max(0.0, safe_distance - d_aa) if d_aa < safe_distance else d_al * 0.5
        typical_collision = max(0.0, (collision_threshold - d_aa) / collision_threshold) if d_aa < collision_threshold else 0.3
        typical_redundancy = 0.3 if M <= N else 0.0  # 启发式：agent 多于 target 时可能冗余

        # 避免除零
        eps = 1e-6

        # 3. 缩放 lambda 使各奖励项初始量级对齐
        cal_s_max = lambda_s_max * (target_magnitude / max(typical_safety, eps))
        cal_r_max = lambda_r_max * (target_magnitude / max(typical_redundancy, eps))
        cal_a = lambda_a * (target_magnitude / max(typical_assign, eps))
        cal_c_max = lambda_c_max * (target_magnitude / max(typical_collision, eps))

        # 4. 限制缩放倍数在合理范围 [0.1, 10.0]，防止极端情况
        def clamp_scale(val, ref):
            ratio = val / ref if ref > 0 else 1.0
            if ratio > 10.0:
                return ref * 10.0
            if ratio < 0.1:
                return ref * 0.1
            return val

        cal_s_max = clamp_scale(cal_s_max, lambda_s_max)
        cal_r_max = clamp_scale(cal_r_max, lambda_r_max)
        cal_a = clamp_scale(cal_a, lambda_a)
        cal_c_max = clamp_scale(cal_c_max, lambda_c_max)

        logger_ws.info(
            "[WeightScheduler] 标定权重: λ_a=%.3f→%.3f, λ_s_max=%.3f→%.3f, "
            "λ_r_max=%.3f→%.3f, λ_c_max=%.3f→%.3f",
            lambda_a, cal_a, lambda_s_max, cal_s_max,
            lambda_r_max, cal_r_max, lambda_c_max, cal_c_max,
        )

        return {
            'lambda_s_max': round(cal_s_max, 4),
            'lambda_r_max': round(cal_r_max, 4),
            'lambda_a': round(cal_a, 4),
            'lambda_c_max': round(cal_c_max, 4),
        }
