"""奖励塑形模块单元测试。"""

import numpy as np
import pytest
from rewards.assignment import AssignmentReward
from rewards.redundancy import RedundancyPenalty
from rewards.safety import SafetyPenalty
from rewards.weight_scheduler import WeightScheduler


class TestAssignmentReward:
    """目标分配引导奖励测试。"""

    @pytest.fixture
    def assigner(self):
        ar = AssignmentReward(lambda_a=0.5, hysteresis_epsilon=0.05, lock_steps=3)
        yield ar
        ar.reset()

    def test_one_to_one_mapping(self, assigner):
        """验证一一映射分配。"""
        agents = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        landmarks = np.array([[0.05, 0.05], [0.55, 0.55], [0.95, 0.95]])
        rewards, info = assigner.compute(agents, landmarks)
        assignment = info['assignment']
        # 每个目标点被分配给一个智能体，每个智能体最多一个目标点
        assert len(assignment) == 3
        assigned_agents = set(assignment.values())
        assert len(assigned_agents) == 3

    def test_distance_matrix(self, assigner):
        """验证距离矩阵计算。"""
        agents = np.array([[0.0, 0.0]])
        landmarks = np.array([[3.0, 4.0]])
        assigner.reset()
        rewards, info = assigner.compute(agents, landmarks)
        assert np.allclose(info['distances'][0, 0], 5.0)

    def test_lock_first_k_steps(self, assigner):
        """验证前 K 步锁定分配。"""
        agents = np.array([[0.0, 0.0], [1.0, 1.0]])
        landmarks = np.array([[0.1, 0.1], [0.9, 0.9]])

        # 第一步（首次分配）
        _, info1 = assigner.compute(agents, landmarks)
        assign1 = dict(info1['assignment'])

        # 第 2-3 步应保持相同分配（lock_steps=3，但首次不计入）
        for step in range(3):
            _, info = assigner.compute(agents, landmarks)
            assert info['assignment'] == assign1

    def test_reward_non_positive(self, assigner):
        """验证分配奖励为非正值（距离惩罚）。"""
        agents = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        landmarks = np.array([[0.1, 0.1], [0.6, 0.6], [1.1, 1.1]])
        rewards, _ = assigner.compute(agents, landmarks)
        for v in rewards.values():
            assert v <= 0.0


class TestRedundancyPenalty:
    """冗余覆盖惩罚测试。"""

    def test_no_redundancy(self):
        """每个目标只有 1 个智能体覆盖 → 惩罚为 0。"""
        rp = RedundancyPenalty(coverage_radius=0.3, lambda_r_max=0.5)
        agents = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        landmarks = np.array([[0.05, 0.05], [0.55, 0.55], [1.05, 1.05]])
        rewards, info = rp.compute(agents, landmarks)
        assert info['redundant_count'] == 0
        assert sum(rewards.values()) == 0.0

    def test_all_agents_same_landmark(self):
        """3 个智能体覆盖同一目标 → penalty = -lambda * 2。"""
        rp = RedundancyPenalty(coverage_radius=0.5, lambda_r_max=0.5)
        agents = np.array([[0.0, 0.0], [0.05, 0.05], [0.1, 0.1]])
        landmarks = np.array([[0.05, 0.05]])
        rewards, info = rp.compute(agents, landmarks)
        assert info['redundant_count'] == 2  # 3-1
        assert abs(info['total_penalty'] + 1.0) < 0.01  # -0.5 * 2 = -1.0

    def test_set_lambda(self):
        """测试动态调整 lambda。"""
        rp = RedundancyPenalty(coverage_radius=0.5, lambda_r_max=0.5)
        agents = np.array([[0.0, 0.0], [0.05, 0.05]])
        landmarks = np.array([[0.025, 0.025]])

        _, info1 = rp.compute(agents, landmarks)
        rp.set_lambda(0.1)
        _, info2 = rp.compute(agents, landmarks)
        # 降低 lambda 后惩罚绝对值减小
        assert abs(info2['total_penalty']) < abs(info1['total_penalty'])


class TestSafetyPenalty:
    """安全距离约束测试。"""

    def test_no_violation(self):
        """所有智能体间距 > d_safe → 惩罚为 0。"""
        sp = SafetyPenalty(safe_distance=0.3, lambda_s_max=0.5)
        agents = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        rewards, info = sp.compute(agents)
        assert info['violations'] == 0
        assert info['total_penalty'] == 0.0

    def test_violation_penalty(self):
        """2 个智能体距离 = d_safe/2 → 验证渐进惩罚。"""
        sp = SafetyPenalty(safe_distance=1.0, lambda_s_max=1.0)
        agents = np.array([[0.0, 0.0], [0.5, 0.0]])  # distance = 0.5
        rewards, info = sp.compute(agents)
        # penalty = 1.0 * (1.0 - 0.5) = 0.5 per violation, total = -0.5
        assert abs(info['total_penalty'] + 0.5) < 0.01

    def test_monotonicity(self):
        """距离越近惩罚越大。"""
        sp = SafetyPenalty(safe_distance=1.0, lambda_s_max=1.0)
        agents_far = np.array([[0.0, 0.0], [0.8, 0.0]])
        agents_near = np.array([[0.0, 0.0], [0.2, 0.0]])

        _, info_far = sp.compute(agents_far)
        _, info_near = sp.compute(agents_near)
        assert abs(info_near['total_penalty']) > abs(info_far['total_penalty'])


class TestWeightScheduler:
    """动态权重调度器测试。"""

    def test_initial_weights(self):
        """Episode 0 时应为早期恒定初始值。"""
        scheduler = WeightScheduler(total_episodes=10000)
        weights = scheduler.get_weights(0, 0.5, 0.3, 0.5, lambda_c_max=0.5)
        assert weights['lambda_a'] == 0.5
        assert weights['lambda_s'] == 0.1  # 0.2 * 0.5
        assert weights['lambda_r'] == 0.03  # 0.1 * 0.3
        assert weights['lambda_c'] == 0.15  # 0.3 * 0.5

    def test_middle_weights(self):
        """50% 时应为中期值。"""
        scheduler = WeightScheduler(total_episodes=10000, early_ratio=0.3)
        mid_ep = 5000  # 50%
        weights = scheduler.get_weights(mid_ep, 0.5, 0.3, 0.5)
        assert weights['lambda_a'] == 0.5  # 不变
        # mid t = (5000-3000)/4000 = 0.5, lambda_s = 0.5*(0.2+0.3*0.5)=0.175
        assert abs(weights['lambda_s'] - 0.175) < 0.01

    def test_final_weights(self):
        """最后一期应为最大值。"""
        scheduler = WeightScheduler(total_episodes=10000)
        weights = scheduler.get_weights(9999, 0.5, 0.3, 0.5)
        assert abs(weights['lambda_s'] - 0.5) < 0.01
        assert abs(weights['lambda_r'] - 0.3) < 0.01

    def test_monotonic_increase(self):
        """权重应单调不减。"""
        scheduler = WeightScheduler(total_episodes=10000)
        prev_s = 0
        prev_r = 0
        for ep in range(0, 10000, 500):
            weights = scheduler.get_weights(ep, 0.5, 0.3, 0.5)
            assert weights['lambda_s'] >= prev_s - 1e-6
            assert weights['lambda_r'] >= prev_r - 1e-6
            prev_s = weights['lambda_s']
            prev_r = weights['lambda_r']

    def test_stage_names(self):
        """验证阶段名称。"""
        scheduler = WeightScheduler(total_episodes=10000, early_ratio=0.3)
        assert scheduler.get_stage(0) == 'early'
        assert scheduler.get_stage(5000) == 'middle'
        assert scheduler.get_stage(9000) == 'late'
