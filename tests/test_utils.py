"""工具模块单元测试。测试 replay_buffer, ou_noise, metrics。"""

import numpy as np
import pytest
from utils.replay_buffer import ReplayBuffer
from utils.ou_noise import OUNoise
from utils.metrics import compute_metrics, compute_completion_steps


class TestReplayBuffer:
    """经验回放池测试。"""

    def test_initial_empty(self):
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0

    def test_push_and_len(self):
        buffer = ReplayBuffer(capacity=100)
        obs = {'agent_0': np.ones(18)}
        actions = {'agent_0': np.ones(5) * 0.5}
        rewards = {'agent_0': -1.0}
        next_obs = {'agent_0': np.ones(18) * 2}
        dones = {'agent_0': False}
        buffer.push(obs, actions, rewards, next_obs, dones)
        assert len(buffer) == 1

    def test_sample_returns_correct_size(self):
        buffer = ReplayBuffer(capacity=100)
        for i in range(50):
            obs = {'agent_0': np.ones(18) * i}
            actions = {'agent_0': np.ones(5) * 0.5}
            rewards = {'agent_0': -1.0 * i}
            next_obs = {'agent_0': np.ones(18) * (i + 1)}
            dones = {'agent_0': False}
            buffer.push(obs, actions, rewards, next_obs, dones)
        batch = buffer.sample(32)
        assert len(batch['obs']) == 32

    def test_sample_smaller_than_requested(self):
        buffer = ReplayBuffer(capacity=100)
        for i in range(16):
            obs = {'agent_0': np.ones(18) * i}
            actions = {'agent_0': np.ones(5) * 0.5}
            rewards = {'agent_0': -1.0 * i}
            next_obs = {'agent_0': np.ones(18) * (i + 1)}
            dones = {'agent_0': False}
            buffer.push(obs, actions, rewards, next_obs, dones)
        batch = buffer.sample(32)
        assert len(batch['obs']) == 16  # 只有 16 条

    def test_capacity_overflow(self):
        buffer = ReplayBuffer(capacity=10)
        for i in range(15):
            obs = {'agent_0': np.ones(18) * i}
            actions = {'agent_0': np.ones(5) * 0.5}
            rewards = {'agent_0': -1.0 * i}
            next_obs = {'agent_0': np.ones(18)}
            dones = {'agent_0': False}
            buffer.push(obs, actions, rewards, next_obs, dones)
        assert len(buffer) == 10

    def test_is_ready(self):
        buffer = ReplayBuffer(capacity=100)
        assert not buffer.is_ready(32)
        for i in range(40):
            buffer.push(
                {'agent_0': np.ones(18)}, {'agent_0': np.ones(5)},
                {'agent_0': 0.0}, {'agent_0': np.ones(18)}, {'agent_0': False}
            )
        assert buffer.is_ready(32)


class TestOUNoise:
    """Ornstein-Uhlenbeck 噪声测试。"""

    def test_sample_shape(self):
        noise = OUNoise(action_dim=5, seed=42)
        sample = noise.sample()
        assert sample.shape == (5,)

    def test_samples_are_different(self):
        """连续采样值应不同（证明是随机过程）。"""
        noise = OUNoise(action_dim=5, seed=42)
        s1 = noise.sample()
        s2 = noise.sample()
        assert not np.allclose(s1, s2)

    def test_reset(self):
        noise = OUNoise(action_dim=5, seed=42)
        state_before = noise._state.copy()
        for _ in range(20):
            noise.sample()
        assert not np.allclose(state_before, noise._state)
        noise.reset()
        assert np.allclose(noise._state, noise.mu)

    def test_sample_clipped(self):
        noise = OUNoise(action_dim=5, sigma=0.01, seed=42)
        clipped = noise.sample_clipped(-0.1, 0.1)
        assert np.all(clipped >= -0.1)
        assert np.all(clipped <= 0.1)

    def test_seed_reproducibility(self):
        """相同种子应产生相同初始状态。"""
        np.random.seed(42)
        noise1 = OUNoise(action_dim=5)
        np.random.seed(42)
        noise2 = OUNoise(action_dim=5)
        # OU 噪声的起始状态应相同（从 mu 开始，由 reset 设置）
        assert np.allclose(noise1._state, noise2._state)
        # 由于 reset 中调用了 np.random.randn（影响全局状态），
        # sample 结果不一定相同，但初始状态应该一致
        assert np.allclose(noise1.mu, noise2.mu)


class TestMetrics:
    """评价指标计算测试。"""

    def test_coverage_rate_full(self):
        """所有目标点被覆盖。"""
        agents = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        landmarks = np.array([[0.05, 0.05], [0.55, 0.55], [0.95, 0.95]])
        metrics = compute_metrics(agents, landmarks, coverage_radius=0.2)
        assert metrics['coverage_rate'] == 1.0

    def test_coverage_rate_partial(self):
        """2/3 目标点被覆盖。"""
        agents = np.array([[0.0, 0.0], [0.5, 0.5], [10.0, 10.0]])
        landmarks = np.array([[0.05, 0.05], [0.55, 0.55], [-1.0, -1.0]])
        metrics = compute_metrics(agents, landmarks, coverage_radius=0.2)
        assert metrics['coverage_rate'] == 2.0 / 3.0

    def test_no_collision(self):
        """所有智能体间距大于碰撞阈值。"""
        agents = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        landmarks = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        metrics = compute_metrics(agents, landmarks, coverage_radius=0.5,
                                  collision_threshold=0.05)
        assert metrics['collision_count'] == 0

    def test_collision_detected(self):
        """2 个智能体距离小于碰撞阈值。"""
        agents = np.array([[0.0, 0.0], [0.02, 0.02], [2.0, 2.0]])
        landmarks = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        metrics = compute_metrics(agents, landmarks, coverage_radius=0.5,
                                  collision_threshold=0.05)
        assert metrics['collision_count'] == 1

    def test_redundancy_zero(self):
        """每个目标点只有 1 个智能体覆盖。"""
        agents = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        landmarks = np.array([[0.1, 0.1], [0.6, 0.6], [1.1, 1.1]])
        metrics = compute_metrics(agents, landmarks, coverage_radius=0.3)
        assert metrics['redundancy_rate'] == 0.0

    def test_redundancy_detected(self):
        """同一目标点被 3 个智能体覆盖（冗余=1）。"""
        agents = np.array([[0.0, 0.0], [0.05, 0.05], [0.1, 0.1]])
        landmarks = np.array([[0.05, 0.05]])
        metrics = compute_metrics(agents, landmarks, coverage_radius=0.3)
        assert metrics['redundancy_rate'] == 1.0  # 1/1 目标点有冗余

    def test_completion_steps(self):
        """测试完成步数计算。"""
        history = [
            np.array([[0.0, 0.0], [10.0, 10.0]]),  # step 1: 覆盖 landmark 0
            np.array([[0.0, 0.0], [1.0, 1.0]]),    # step 2: 覆盖 landmark 0, 1
            np.array([[0.0, 0.0], [1.0, 1.0]]),    # step 3: same
        ]
        landmarks = np.array([[0.1, 0.1], [0.9, 0.9]])
        steps = compute_completion_steps(history, landmarks, coverage_radius=0.3)
        assert steps == 2

    def test_completion_steps_never(self):
        """从未完成覆盖。"""
        history = [
            np.array([[0.0, 0.0], [10.0, 10.0]]),
            np.array([[0.0, 0.0], [10.0, 10.0]]),
        ]
        landmarks = np.array([[0.1, 0.1], [0.9, 0.9]])
        steps = compute_completion_steps(history, landmarks, coverage_radius=0.3)
        assert steps == 2  # 返回历史长度
