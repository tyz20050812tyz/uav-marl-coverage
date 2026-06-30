"""网络模块单元测试。"""

import numpy as np
import torch
import pytest
from networks.actor import Actor
from networks.critic import Critic, TwinCritic


class TestActor:
    """Actor 网络测试。"""

    def test_forward_shape(self):
        """测试前向传播输出 shape。"""
        actor = Actor(obs_dim=18, act_dim=5)
        x = torch.randn(32, 18)
        out = actor(x)
        assert out.shape == (32, 5)

    def test_output_in_range(self):
        """测试 Sigmoid 确保输出 ∈ [0, 1]。"""
        actor = Actor(obs_dim=18)
        x = torch.randn(100, 18)
        out = actor(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_get_action_shape(self):
        """测试 get_action 输出 shape。"""
        actor = Actor(obs_dim=18)
        obs = torch.randn(18)
        action = actor.get_action(obs)
        assert action.shape == (5,)

    def test_get_action_with_noise(self):
        """测试加噪声后仍 clip 到 [0, 1]。"""
        actor = Actor(obs_dim=18)
        obs = torch.randn(18)
        noise = np.ones(5) * 0.5  # 大噪声
        action = actor.get_action(obs, noise=noise)
        assert action.shape == (5,)
        assert (action >= 0.0).all()
        assert (action <= 1.0).all()

    def test_variable_obs_dim(self):
        """测试不同 obs_dim（N=3/4/5）的网络均可 forward。"""
        for obs_dim in [18, 24, 30]:  # 不同 N 的观测维度可能不同
            actor = Actor(obs_dim=obs_dim)
            x = torch.randn(4, obs_dim)
            out = actor(x)
            assert out.shape == (4, 5)

    def test_deterministic_action(self):
        """测试确定性动作无噪声且可复现。"""
        actor = Actor(obs_dim=18)
        obs = torch.ones(18)
        a1 = actor.get_deterministic_action(obs)
        a2 = actor.get_deterministic_action(obs)
        assert np.allclose(a1, a2)

    def test_parameters_not_nan(self):
        """测试初始化后参数无 NaN。"""
        actor = Actor(obs_dim=18)
        for name, param in actor.named_parameters():
            assert not torch.isnan(param).any(), f"{name} contains NaN"


class TestCritic:
    """Critic 网络测试。"""

    def test_forward_shape(self):
        """测试前向传播输出 shape。"""
        critic = Critic(all_obs_dim=18 * 3, all_act_dim=5 * 3)
        all_obs = torch.randn(32, 18 * 3)
        all_actions = torch.randn(32, 5 * 3)
        q = critic(all_obs, all_actions)
        assert q.shape == (32, 1)

    def test_variable_N(self):
        """测试不同 N 对应的输入维度。"""
        for N in [3, 4, 5]:
            all_obs_dim = 18 * N  # 每条 18 dim
            all_act_dim = 5 * N
            critic = Critic(all_obs_dim=all_obs_dim, all_act_dim=all_act_dim)
            all_obs = torch.randn(4, all_obs_dim)
            all_actions = torch.randn(4, all_act_dim)
            q = critic(all_obs, all_actions)
            assert q.shape == (4, 1)


class TestTwinCritic:
    """双 Critic 网络测试。"""

    def test_two_critics_independent(self):
        """测试两个 Critic 参数独立。"""
        twin = TwinCritic(all_obs_dim=18 * 3, all_act_dim=5 * 3)
        params_a = list(twin.critic_A.parameters())
        params_b = list(twin.critic_B.parameters())
        # 两个 critic 的参数不应共享内存
        for pa, pb in zip(params_a, params_b):
            assert pa.data_ptr() != pb.data_ptr()

    def test_q_values_shape(self):
        """测试 q_values 返回 shape。"""
        twin = TwinCritic(all_obs_dim=54, all_act_dim=15)
        all_obs = torch.randn(8, 54)
        all_actions = torch.randn(8, 15)
        qa, qb = twin.q_values(all_obs, all_actions)
        assert qa.shape == (8, 1)
        assert qb.shape == (8, 1)

    def test_q_min(self):
        """测试 q_min 确实是 min(Q_A, Q_B)。"""
        twin = TwinCritic(all_obs_dim=54, all_act_dim=15)
        all_obs = torch.randn(4, 54)
        all_actions = torch.randn(4, 15)
        qa, qb = twin.q_values(all_obs, all_actions)
        q_min = twin.q_min(all_obs, all_actions)
        assert torch.all(q_min <= qa + 1e-6)
        assert torch.all(q_min <= qb + 1e-6)

    def test_forward_is_critic_A(self):
        """测试 forward 等同于 critic_A forward。"""
        twin = TwinCritic(all_obs_dim=54, all_act_dim=15)
        all_obs = torch.randn(2, 54)
        all_actions = torch.randn(2, 15)
        fwd = twin(all_obs, all_actions)
        qa, _ = twin.q_values(all_obs, all_actions)
        assert torch.allclose(fwd, qa)
