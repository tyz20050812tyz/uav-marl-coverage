"""环境封装模块单元测试。"""

import numpy as np
import pytest
from env.simple_spread_wrapper import SimpleSpreadWrapper


class TestSimpleSpreadWrapper:
    """simple_spread_v3 环境封装测试。"""

    @pytest.fixture
    def env_n3(self):
        """N=3 默认环境。"""
        env = SimpleSpreadWrapper(num_agents=3)
        yield env
        env.close()

    def test_obs_dim(self, env_n3):
        """测试 obs_dim 返回值正确。"""
        assert env_n3.obs_dim == 18, f"Expected 18, got {env_n3.obs_dim}"

    def test_act_dim(self, env_n3):
        """测试 act_dim 返回 5。"""
        assert env_n3.act_dim == 5, f"Expected 5, got {env_n3.act_dim}"

    def test_num_agents(self, env_n3):
        """测试 agent_names 数量正确。"""
        assert len(env_n3.agent_names) == 3

    def test_world_size(self, env_n3):
        """测试 world_size 为 2.0。"""
        assert env_n3.world_size == 2.0

    def test_reset_returns_dict(self, env_n3):
        """测试 reset 返回 dict 且包含 3 个 agent。"""
        obs, info = env_n3.reset()
        assert len(obs) == 3
        for name in env_n3.agent_names:
            assert name in obs

    def test_obs_shape_per_agent(self, env_n3):
        """测试每个 agent 的观测 shape 正确。"""
        obs, _ = env_n3.reset()
        for name in env_n3.agent_names:
            assert obs[name].shape == (18,)

    def test_action_space_continuous(self, env_n3):
        """测试连续动作空间为 Box(0, 1, (5,))。"""
        _, _ = env_n3.reset()
        import gymnasium as gym
        act_space = env_n3._env.action_space(env_n3.agent_names[0])
        assert isinstance(act_space, gym.spaces.Box)
        assert act_space.shape == (5,)
        assert act_space.low[0] == 0.0
        assert act_space.high[0] == 1.0

    def test_step_runs_no_error(self, env_n3):
        """测试 step 无异常。"""
        obs, _ = env_n3.reset()
        actions = {k: np.ones(5) * 0.5 for k in obs.keys()}
        obs2, rewards, terms, truncs, infos = env_n3.step(actions)
        assert len(obs2) == 3
        assert len(rewards) == 3
        assert len(terms) == 3
        assert len(truncs) == 3

    def test_full_episode(self, env_n3):
        """测试完整 episode 运行无异常。"""
        np.random.seed(42)
        obs, _ = env_n3.reset()
        total_reward = 0.0
        step_count = 0
        done = False
        while not done and step_count < env_n3.max_cycles:
            actions = {
                k: np.random.uniform(0, 1, size=5)
                for k in obs.keys()
            }
            obs, rewards, terms, truncs, _ = env_n3.step(actions)
            total_reward += sum(rewards.values())
            step_count += 1
            done = all(terms.values()) or all(truncs.values())
        assert step_count > 0
        assert not np.isnan(total_reward)
        assert not np.isinf(total_reward)

    def test_render(self, env_n3):
        """测试 render 返回正确 shape。"""
        env_n3.reset()
        frame = env_n3.render()
        assert frame.shape == (700, 700, 3)
        assert frame.dtype == np.uint8
        assert frame.min() >= 0 and frame.max() <= 255

    def test_multiple_resets(self, env_n3):
        """测试同一环境多次 reset 无异常且状态有变化。"""
        obs1, _ = env_n3.reset()
        obs2, _ = env_n3.reset()
        # 不同 reset 的观测应不同（环境随机初始化）
        all_same = True
        for k in env_n3.agent_names:
            if not np.allclose(obs1[k], obs2[k]):
                all_same = False
                break
        assert not all_same, "Expected different observations after reset"

    def test_set_config(self, env_n3):
        """测试 set_config 切换智能体数量。"""
        env_n3.set_config(num_agents=4, max_cycles=70)
        assert env_n3.num_agents == 4
        assert env_n3.max_cycles == 70
        assert len(env_n3.agent_names) == 4
        obs, _ = env_n3.reset()
        assert len(obs) == 4

    def test_variable_N(self):
        """测试不同 N 值均可正常初始化。"""
        for n in [3, 4, 5]:
            env = SimpleSpreadWrapper(num_agents=n)
            assert env.num_agents == n
            assert len(env.agent_names) == n
            obs, _ = env.reset()
            assert len(obs) == n
            env.close()

    def test_get_world_state(self, env_n3):
        """测试 get_world_state 返回正确的状态字典。"""
        env_n3.reset()
        state = env_n3.get_world_state()
        assert 'agent_positions' in state
        assert 'landmark_positions' in state
        assert 'agent_velocities' in state
        assert state['agent_positions'].shape == (3, 2)
        assert state['landmark_positions'].shape == (3, 2)
        assert state['agent_velocities'].shape == (3, 2)
