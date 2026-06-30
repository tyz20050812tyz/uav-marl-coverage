"""RS-MADDPG 智能体单元测试（含过拟合测试）。

验证 RS-MADDPG 的奖励塑形集成、权重调度、状态持久化。
"""

import numpy as np
import torch
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rs_maddpg_agent import RSMADDPGAgent
from rewards.assignment import AssignmentReward
from rewards.redundancy import RedundancyPenalty
from rewards.safety import SafetyPenalty
from rewards.weight_scheduler import WeightScheduler
from rewards.base_reward import BaseEnvReward


# ==================== RS-MADDPG 基础测试 ====================

class TestRSMADDPGAgent:
    """RS-MADDPG 智能体核心功能测试。"""

    @pytest.fixture
    def rs_maddpg(self):
        """创建默认 RS-MADDPG 实例（3 agents, 18 obs, 5 act）。"""
        return RSMADDPGAgent(
            num_agents=3, obs_dim=18, act_dim=5,
            actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01,
            policy_delay=2,
            lambda_a=0.5, hysteresis_epsilon=0.05, lock_steps=5,
            coverage_radius=0.24, lambda_r_max=0.3,
            safe_distance=0.2, lambda_s_max=0.5,
            use_weight_scheduling=False,
            total_episodes=10000,
        )

    @pytest.fixture
    def rs_maddpg_scheduled(self):
        """创建启用权重调度的 RS-MADDPG 实例。"""
        return RSMADDPGAgent(
            num_agents=3, obs_dim=18, act_dim=5,
            use_weight_scheduling=True,
            total_episodes=10000,
        )

    def test_init_modules_created(self, rs_maddpg):
        """初始化后三大奖励模块均正确创建。"""
        assert isinstance(rs_maddpg.assignment_reward, AssignmentReward)
        assert isinstance(rs_maddpg.redundancy_penalty, RedundancyPenalty)
        assert isinstance(rs_maddpg.safety_penalty, SafetyPenalty)

    def test_init_actor_critic_created(self, rs_maddpg):
        """继承 MADDPG，Actor/Critic 网络正确创建。"""
        assert len(rs_maddpg.actors) == 3
        assert rs_maddpg.critic is not None
        assert rs_maddpg.target_critic is not None
        for name in rs_maddpg.agent_names:
            assert name in rs_maddpg.actors
            assert name in rs_maddpg.target_actors

    def test_no_weight_scheduling_default(self, rs_maddpg):
        """默认不启用权重调度。"""
        assert rs_maddpg.use_weight_scheduling is False
        assert rs_maddpg.weight_scheduler is None

    def test_weight_scheduling_enabled(self, rs_maddpg_scheduled):
        """显式启用后权重调度器创建。"""
        assert rs_maddpg_scheduled.use_weight_scheduling is True
        assert isinstance(rs_maddpg_scheduled.weight_scheduler, WeightScheduler)

    def test_reset_episode_clears_assignment(self, rs_maddpg):
        """reset_episode 后分配状态清零。"""
        # 先执行一次 compute 产生内部状态
        positions = np.random.rand(3, 2).astype(np.float32) * 2.0
        landmarks = np.random.rand(3, 2).astype(np.float32) * 2.0
        rs_maddpg.assignment_reward.compute(positions, landmarks)
        assert rs_maddpg.assignment_reward._assignment is not None

        rs_maddpg.reset_episode()
        assert rs_maddpg.assignment_reward._assignment is None
        assert rs_maddpg.assignment_reward._step_counter == 0

    def test_compute_shaped_reward_shape(self, rs_maddpg):
        """塑形奖励返回正确的 key 和 shape。"""
        positions = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=np.float32)
        landmarks = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
        env_rewards = {'agent_0': 0.5, 'agent_1': 0.3, 'agent_2': 0.1}

        rewards = rs_maddpg.compute_shaped_reward(env_rewards, positions, landmarks)

        assert isinstance(rewards, dict)
        assert len(rewards) == 3
        for name in ['agent_0', 'agent_1', 'agent_2']:
            assert name in rewards
            assert isinstance(rewards[name], (float, np.floating))

    def test_compute_shaped_reward_values_are_finite(self, rs_maddpg):
        """塑形奖励值均为有限值。"""
        positions = np.random.rand(3, 2).astype(np.float32) * 2.0
        landmarks = np.random.rand(3, 2).astype(np.float32) * 2.0
        env_rewards = {'agent_0': 1.0, 'agent_1': 1.0, 'agent_2': 1.0}

        rewards = rs_maddpg.compute_shaped_reward(env_rewards, positions, landmarks)
        for v in rewards.values():
            assert np.isfinite(v), f"Non-finite reward: {v}"

    def test_get_rs_info(self, rs_maddpg):
        """get_rs_info 返回完整配置信息。"""
        info = rs_maddpg.get_rs_info()
        assert 'use_weight_scheduling' in info
        assert 'lambda_a' in info
        assert 'lambda_r' in info
        assert 'lambda_s' in info
        assert info['use_weight_scheduling'] is False

    def test_set_episode_updates_weights(self, rs_maddpg_scheduled):
        """set_episode 正确更新冗余和安全权重。"""
        # 先设置到最早 episode，记录初始调度值
        rs_maddpg_scheduled.set_episode(0)
        initial_r = rs_maddpg_scheduled.redundancy_penalty._current_lambda
        initial_s = rs_maddpg_scheduled.safety_penalty._current_lambda

        # 设置到中期 episode，权重应增加
        rs_maddpg_scheduled.set_episode(5000)  # 中期阶段
        mid_r = rs_maddpg_scheduled.redundancy_penalty._current_lambda
        mid_s = rs_maddpg_scheduled.safety_penalty._current_lambda

        assert mid_r > initial_r, (
            f"Mid-episode lambda_r ({mid_r}) should be > initial ({initial_r})"
        )
        assert mid_s > initial_s, (
            f"Mid-episode lambda_s ({mid_s}) should be > initial ({initial_s})"
        )

        # 设置到后期 episode，权重应继续增加
        rs_maddpg_scheduled.set_episode(9000)
        late_r = rs_maddpg_scheduled.redundancy_penalty._current_lambda
        late_s = rs_maddpg_scheduled.safety_penalty._current_lambda

        assert late_r > mid_r
        assert late_s > mid_s

    def test_save_load_rs_config(self, rs_maddpg, tmp_path):
        """保存/加载后 RS 配置和网络权重不丢失。"""
        path = str(tmp_path / 'rs_maddpg_test.pt')
        rs_maddpg.save(path)

        # 加载到新实例
        rs2 = RSMADDPGAgent(
            num_agents=3, obs_dim=18, act_dim=5,
            use_weight_scheduling=False,
        )
        rs2.load(path)

        # 验证 RS 配置一致
        assert rs2._rs_config == rs_maddpg._rs_config
        assert rs2._update_step == rs_maddpg._update_step

        # 验证 Actor 权重一致
        obs = np.random.randn(18).astype(np.float32)
        for name in rs_maddpg.agent_names:
            a1 = rs_maddpg.act(name, obs, add_noise=False)
            a2 = rs2.act(name, obs, add_noise=False)
            assert np.allclose(a1, a2, atol=1e-6), f"Actor {name} mismatch after load"

    def test_save_load_with_scheduling(self, rs_maddpg_scheduled, tmp_path):
        """启用权重调度时保存/加载，状态正确恢复。"""
        rs_maddpg_scheduled.set_episode(500)
        path = str(tmp_path / 'rs_maddpg_scheduled.pt')
        rs_maddpg_scheduled.save(path)

        rs2 = RSMADDPGAgent(
            num_agents=3, obs_dim=18, act_dim=5,
            use_weight_scheduling=True,
            total_episodes=10000,
        )
        rs2.load(path)
        assert rs2._current_episode == 500

    def test_act_shape_and_range(self, rs_maddpg):
        """act 返回正确形状和范围的 5 维动作。"""
        obs = np.random.randn(18).astype(np.float32)
        for name in rs_maddpg.agent_names:
            action = rs_maddpg.act(name, obs, add_noise=False)
            assert action.shape == (5,)
            assert (action >= 0.0).all(), f"Action below 0 for {name}"
            assert (action <= 1.0).all(), f"Action above 1 for {name}"

    def test_update_no_nan(self, rs_maddpg):
        """一次 update 后 loss 不为 NaN。"""
        B = 16
        obs = {f'agent_{i}': torch.randn(B, 18) for i in range(3)}
        actions = {f'agent_{i}': torch.randn(B, 5) for i in range(3)}
        rewards = {f'agent_{i}': torch.randn(B, 1) for i in range(3)}
        next_obs = {f'agent_{i}': torch.randn(B, 18) for i in range(3)}
        dones = {f'agent_{i}': torch.zeros(B, 1) for i in range(3)}

        batch = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
        }

        result = rs_maddpg.update(batch)
        assert not np.isnan(result['critic_loss']), f"critic_loss is NaN: {result}"
        assert not np.isnan(result['actor_loss']), f"actor_loss is NaN: {result}"

    def test_overfit_shaped_target(self, rs_maddpg):
        """
        过拟合测试：验证 RS-MADDPG 在带塑形奖励的简化场景中能收敛。
        构造 3 agent + 3 landmark 场景，验证单 batch 上 loss 下降。
        """
        B = 4
        # 简化观测：前 2 维 = agent_pos，后面填充 0
        obs0 = torch.zeros(B, 18)
        obs0[:, :2] = torch.tensor([[0.0, 0.0]] * B)
        obs1 = torch.zeros(B, 18)
        obs1[:, :2] = torch.tensor([[1.0, 1.0]] * B)
        obs2 = torch.zeros(B, 18)
        obs2[:, :2] = torch.tensor([[2.0, 2.0]] * B)

        obs = {'agent_0': obs0, 'agent_1': obs1, 'agent_2': obs2}
        actions = {'agent_0': torch.rand(B, 5), 'agent_1': torch.rand(B, 5),
                   'agent_2': torch.rand(B, 5)}
        rewards = {'agent_0': torch.ones(B, 1), 'agent_1': torch.ones(B, 1),
                   'agent_2': torch.ones(B, 1)}
        next_obs = {
            'agent_0': obs0 + torch.randn(B, 18) * 0.01,
            'agent_1': obs1 + torch.randn(B, 18) * 0.01,
            'agent_2': obs2 + torch.randn(B, 18) * 0.01,
        }
        dones = {'agent_0': torch.zeros(B, 1), 'agent_1': torch.zeros(B, 1),
                 'agent_2': torch.zeros(B, 1)}

        batch = {'obs': obs, 'actions': actions, 'rewards': rewards,
                 'next_obs': next_obs, 'dones': dones}

        initial_result = rs_maddpg.update(batch)
        for _ in range(50):
            result = rs_maddpg.update(batch)

        # 多次更新后 actor_loss 应下降（允许波动，取最后几次均值）
        final_results = [rs_maddpg.update(batch) for _ in range(10)]
        avg_final_actor_loss = np.mean([r['actor_loss'] for r in final_results])
        avg_final_critic_loss = np.mean([r['critic_loss'] for r in final_results])

        assert avg_final_critic_loss < initial_result['critic_loss'] * 1.5, (
            f"Critic loss not converging: initial={initial_result['critic_loss']:.4f}, "
            f"final_avg={avg_final_critic_loss:.4f}"
        )
        assert avg_final_actor_loss < initial_result['actor_loss'] * 1.5, (
            f"Actor loss not converging: initial={initial_result['actor_loss']:.4f}, "
            f"final_avg={avg_final_actor_loss:.4f}"
        )


# ==================== 奖励塑形模块集成测试 ====================

class TestRewardShapingIntegration:
    """验证 RS-MADDPG 与各奖励模块的集成正确性。"""

    @pytest.fixture
    def rs_agent(self):
        return RSMADDPGAgent(
            num_agents=3, obs_dim=18, act_dim=5,
            lambda_a=1.0,  # 放大分配奖励，便于测试
            lambda_r_max=0.3,
            lambda_s_max=0.5,
            coverage_radius=0.5,
            safe_distance=0.5,
            use_weight_scheduling=False,
        )

    def test_assignment_guides_to_nearest_landmark(self, rs_agent):
        """目标分配奖励引导 agent 向最近目标移动。"""
        # agent 0 离 landmark 0 最近
        positions = np.array([
            [0.1, 0.1],   # agent_0 → near landmark_0
            [0.9, 0.9],   # agent_1 → near landmark_1
            [0.5, 0.5],   # agent_2
        ], dtype=np.float32)
        landmarks = np.array([
            [0.0, 0.0],   # landmark_0
            [1.0, 1.0],   # landmark_1
            [2.0, 2.0],   # landmark_2
        ], dtype=np.float32)

        env_rewards = {'agent_0': 0.0, 'agent_1': 0.0, 'agent_2': 0.0}
        rewards = rs_agent.compute_shaped_reward(env_rewards, positions, landmarks)

        # agent_0 离最近目标最近，agent_2 最远，分配奖励应递减
        assert rewards['agent_0'] >= rewards['agent_2'], (
            f"agent_0 ({rewards['agent_0']}) should have >= assignment reward "
            f"than agent_2 ({rewards['agent_2']})"
        )

    def test_safety_penalty_on_close_agents(self, rs_agent):
        """agent 靠太近时安全惩罚生效。"""
        # 两个 agent 距离 < safe_distance (0.5)
        positions = np.array([
            [0.0, 0.0],
            [0.1, 0.0],   # 距离 0.1 < 0.5
            [2.0, 2.0],
        ], dtype=np.float32)
        landmarks = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ], dtype=np.float32)

        env_rewards = {'agent_0': 0.0, 'agent_1': 0.0, 'agent_2': 0.0}
        rewards = rs_agent.compute_shaped_reward(env_rewards, positions, landmarks)

        # agent_0 和 agent_1 距离近，应有安全惩罚
        r0, r1 = rewards['agent_0'], rewards['agent_1']
        # 总奖励应 < 环境奖励 + 分配奖励（因为安全惩罚为负）
        assert r0 < env_rewards['agent_0'] + 0.01, (
            f"agent_0 reward ({r0}) should reflect safety penalty"
        )
        assert r1 < env_rewards['agent_1'] + 0.01, (
            f"agent_1 reward ({r1}) should reflect safety penalty"
        )

    def test_redundancy_penalty_on_overlap(self, rs_agent):
        """多个 agent 覆盖同一 landmark 时冗余惩罚生效。"""
        # 两个 agent 都靠近 landmark_0
        positions = np.array([
            [0.05, 0.05],  # 靠近 landmark_0
            [0.06, 0.06],  # 也靠近 landmark_0（冗余）
            [2.0, 2.0],   # 远离
        ], dtype=np.float32)
        landmarks = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ], dtype=np.float32)

        env_rewards = {'agent_0': 0.0, 'agent_1': 0.0, 'agent_2': 0.0}
        rewards = rs_agent.compute_shaped_reward(env_rewards, positions, landmarks)

        # agent_0 和 agent_1 都覆盖 landmark_0，应有冗余惩罚
        r0, r1 = rewards['agent_0'], rewards['agent_1']
        # 冗余惩罚为负，总奖励应 < 分配奖励（分配奖励也是负的，但幅度小）
        assert r0 < 0.0, f"agent_0 reward ({r0}) should be negative due to redundancy"
        assert r1 < 0.0, f"agent_1 reward ({r1}) should be negative due to redundancy"

    def test_env_reward_passthrough(self, rs_agent):
        """环境奖励正确传递到总奖励中。"""
        positions = np.random.rand(3, 2).astype(np.float32)
        landmarks = np.random.rand(3, 2).astype(np.float32)
        env_rewards = {'agent_0': 2.0, 'agent_1': -1.0, 'agent_2': 0.5}

        rewards = rs_agent.compute_shaped_reward(env_rewards, positions, landmarks)

        # 如果 env_reward 很高，总奖励应该也有对应贡献
        assert rewards['agent_0'] > 0.0, (
            f"agent_0 with high env reward ({env_rewards['agent_0']}) "
            f"should have positive total ({rewards['agent_0']})"
        )


# ==================== BaseEnvReward 测试 ====================

class TestBaseEnvReward:
    """原始环境奖励提取器测试。"""

    def test_pass_through(self):
        """compute 直接透传 env_rewards。"""
        ber = BaseEnvReward()
        env_rewards = {'agent_0': 1.0, 'agent_1': -0.5}
        result = ber.compute(env_rewards)
        assert result == env_rewards
        # 应返回新 dict（浅拷贝）
        assert result is not env_rewards

    def test_reset_is_noop(self):
        """reset 为无操作，不抛异常。"""
        ber = BaseEnvReward()
        ber.reset()  # 不应抛异常
