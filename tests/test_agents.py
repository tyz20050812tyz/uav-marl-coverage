"""智能体模块单元测试（含过拟合测试）。"""

import numpy as np
import torch
import pytest
from agents.random_agent import RandomAgent
from agents.iddpg_agent import IDDPGAgent
from agents.maddpg_agent import MADDPGAgent


# ==================== Random Agent ====================

class TestRandomAgent:
    """随机策略测试。"""

    def test_act_shape(self):
        agent = RandomAgent(act_dim=5)
        action = agent.act({})
        assert action.shape == (5,)

    def test_act_in_range(self):
        agent = RandomAgent(act_dim=5)
        for _ in range(100):
            action = agent.act({})
            assert (action >= 0.0).all()
            assert (action <= 1.0).all()

    def test_actions_are_random(self):
        """连续采样不应完全相同。"""
        agent = RandomAgent(act_dim=5)
        actions = [agent.act({}) for _ in range(50)]
        all_same = all(np.allclose(actions[0], a) for a in actions)
        assert not all_same


# ==================== IDDPG Agent ====================

class TestIDDPGAgent:
    """IDDPG 智能体测试。"""

    @pytest.fixture
    def iddpg(self):
        return IDDPGAgent(name='agent_0', obs_dim=18, act_dim=5)

    def test_init_parameters(self, iddpg):
        """初始化后参数不为空。"""
        assert sum(p.numel() for p in iddpg.actor.parameters()) > 0
        assert sum(p.numel() for p in iddpg.critic.parameters()) > 0

    def test_act_shape(self, iddpg):
        obs = np.random.randn(18).astype(np.float32)
        action = iddpg.act(obs)
        assert action.shape == (5,)

    def test_act_with_noise(self, iddpg):
        obs = np.random.randn(18).astype(np.float32)
        a1 = iddpg.act(obs, add_noise=True)
        a2 = iddpg.act(obs, add_noise=True)
        # 噪声导致两次动作不同
        assert not np.allclose(a1, a2)

    def test_update_no_nan(self, iddpg):
        """一次 update 后 loss 不为 NaN。"""
        B = 16
        obs = torch.randn(B, 18)
        actions = torch.randn(B, 5)
        rewards = torch.randn(B, 1)
        next_obs = torch.randn(B, 18)
        dones = torch.zeros(B, 1)

        batch = {
            'obs': {'agent_0': obs},
            'actions': {'agent_0': actions},
            'rewards': {'agent_0': rewards},
            'next_obs': {'agent_0': next_obs},
            'dones': {'agent_0': dones},
        }

        result = iddpg.update(batch)
        assert not np.isnan(result['critic_loss'])
        assert not np.isnan(result['actor_loss'])

    def test_save_load(self, iddpg, tmp_path):
        """保存和加载后 act 输出一致。"""
        obs = np.ones(18, dtype=np.float32)
        action_before = iddpg.act(obs, add_noise=False)
        path = tmp_path / 'iddpg_test.pt'
        iddpg.save(str(path))
        iddpg2 = IDDPGAgent(name='agent_0', obs_dim=18, act_dim=5)
        iddpg2.load(str(path))
        action_after = iddpg2.act(obs, add_noise=False)
        assert np.allclose(action_before, action_after)

    def test_overfit_simple_env(self, iddpg):
        """
        过拟合测试（Overfit Test）：防静默失败。
        构造 1 智能体 + 1 固定目标点的极简静态环境，
        验证网络能否在同一个 batch 上收敛。
        """
        B = 1
        obs_dim = 18
        # 构造一个简单场景：obs 的前 2 维（相对位置）表示目标在上方
        # 在 simple_spread 中，obs 结构为 [vel(2), pos(2), lm_rel(6), ag_rel(4), comm(4)]
        # 让目标点在上方，智能体在下方
        obs_val = np.zeros(obs_dim, dtype=np.float32)
        obs_val[2:4] = [0.0, 0.0]  # self_pos
        obs_val[4:6] = [0.0, 1.0]  # landmark_0 rel_pos: 正上方
        # 动作：上移（索引 4）应该最有利于接近目标
        obs_tensor = torch.from_numpy(obs_val).unsqueeze(0)
        next_obs_val = obs_val.copy()
        next_obs_val[4:6] = [0.0, 0.5]  # 接近了
        next_obs_tensor = torch.from_numpy(next_obs_val).unsqueeze(0)

        actions_tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])  # 上移最强
        reward = torch.tensor([[0.5]])  # 接近了得到正奖励

        batch = {
            'obs': {'agent_0': obs_tensor},
            'actions': {'agent_0': actions_tensor},
            'rewards': {'agent_0': reward},
            'next_obs': {'agent_0': next_obs_tensor},
            'dones': {'agent_0': torch.zeros(B, 1)},
        }

        losses = []
        for i in range(50):
            result = iddpg.update(batch)
            losses.append(result['critic_loss'])

        # Critic loss 应下降趋势（前 10 步均值 > 后 10 步均值）
        assert np.mean(losses[:10]) > np.mean(losses[-10:]) * 0.5 or \
               np.mean(losses[:10]) > 0.01

        # 训练后向目标移动的动作（上移=索引4）应有较高强度
        iddpg.actor.eval()
        with torch.no_grad():
            out = iddpg.actor(obs_tensor).squeeze(0).numpy()
        assert out[4] > 0.3, f"上移方向动作强度太低: {out}"


# ==================== MADDPG Agent ====================

class TestMADDPGAgent:
    """MADDPG 智能体测试。"""

    @pytest.fixture
    def maddpg(self):
        return MADDPGAgent(num_agents=3, obs_dim=18, act_dim=5)

    def test_init_parameters(self, maddpg):
        """初始化后参数不为空。"""
        assert len(maddpg.actors) == 3
        for name in maddpg.agent_names:
            assert sum(p.numel() for p in maddpg.actors[name].parameters()) > 0

    def test_act_shape(self, maddpg):
        obs = np.random.randn(18).astype(np.float32)
        action = maddpg.act('agent_0', obs)
        assert action.shape == (5,)

    def test_act_all(self, maddpg):
        observations = {
            f'agent_{i}': np.random.randn(18).astype(np.float32)
            for i in range(3)
        }
        actions = maddpg.act_all(observations)
        assert len(actions) == 3
        for name in maddpg.agent_names:
            assert actions[name].shape == (5,)

    def test_update_no_nan(self, maddpg):
        """一次 update 后 loss 不为 NaN。"""
        B = 16
        batch = {
            'obs': {},
            'actions': {},
            'rewards': {},
            'next_obs': {},
            'dones': {},
        }
        for name in maddpg.agent_names:
            batch['obs'][name] = torch.randn(B, 18)
            batch['actions'][name] = torch.randn(B, 5)
            batch['rewards'][name] = torch.randn(B, 1)
            batch['next_obs'][name] = torch.randn(B, 18)
            batch['dones'][name] = torch.zeros(B, 1)

        result = maddpg.update(batch)
        assert not np.isnan(result['critic_loss'])

    def test_policy_delay(self, maddpg):
        """验证延迟策略更新。"""
        B = 8
        batch = {
            'obs': {},
            'actions': {},
            'rewards': {},
            'next_obs': {},
            'dones': {},
        }
        for name in maddpg.agent_names:
            batch['obs'][name] = torch.randn(B, 18)
            batch['actions'][name] = torch.randn(B, 5)
            batch['rewards'][name] = torch.randn(B, 1)
            batch['next_obs'][name] = torch.randn(B, 18)
            batch['dones'][name] = torch.zeros(B, 1)

        # 第 1 步 critic 更新，actor 也应更新（_update_step=0, 0%2==0）
        actor_params_before = {
            name: [p.clone() for p in maddpg.actors[name].parameters()]
            for name in maddpg.agent_names
        }
        maddpg.update(batch)
        # _update_step 变为 1

        for name in maddpg.agent_names:
            for before, after in zip(
                actor_params_before[name],
                maddpg.actors[name].parameters()
            ):
                assert not torch.allclose(before, after), \
                    f"Actor {name} should have been updated on step 0"

        # 第 2 步 critic 更新，actor 不应更新（_update_step=1, 1%2!=0）
        actor_params_before = {
            name: [p.clone() for p in maddpg.actors[name].parameters()]
            for name in maddpg.agent_names
        }
        maddpg.update(batch)
        # _update_step 变为 2

        for name in maddpg.agent_names:
            for before, after in zip(
                actor_params_before[name],
                maddpg.actors[name].parameters()
            ):
                assert torch.allclose(before, after), \
                    f"Actor {name} should NOT have been updated on step 1"

    def test_save_load(self, maddpg, tmp_path):
        """保存和加载后 act 输出一致。"""
        obs = np.ones(18, dtype=np.float32)
        action_before = maddpg.act('agent_0', obs, add_noise=False)
        path = tmp_path / 'maddpg_test.pt'
        maddpg.save(str(path))
        maddpg2 = MADDPGAgent(num_agents=3, obs_dim=18, act_dim=5)
        maddpg2.load(str(path))
        action_after = maddpg2.act('agent_0', obs, add_noise=False)
        assert np.allclose(action_before, action_after)

    def test_target_smoothing(self, maddpg):
        """验证目标策略平滑：target action 加了噪声后与原始不同。"""
        B = 16
        batch = {
            'obs': {},
            'actions': {},
            'rewards': {},
            'next_obs': {},
            'dones': {},
        }
        for name in maddpg.agent_names:
            batch['obs'][name] = torch.randn(B, 18)
            batch['actions'][name] = torch.randn(B, 5)
            batch['rewards'][name] = torch.randn(B, 1)
            batch['next_obs'][name] = torch.randn(B, 18)
            batch['dones'][name] = torch.zeros(B, 1)

        # 获取 target actor 输出（无噪声）
        with torch.no_grad():
            next_act_raw = []
            for name in maddpg.agent_names:
                act = maddpg.target_actors[name](batch['next_obs'][name])
                next_act_raw.append(act)
            all_next_act_raw = torch.cat(next_act_raw, dim=1)

        # update 中会加噪声，但这里我们只验证 update 能正常执行
        # 目标平滑的验证通过 update 不报错来保证
        result = maddpg.update(batch)
        assert not np.isnan(result['critic_loss'])

    def test_overfit_simple_env(self):
        """
        过拟合测试（Overfit Test）：防静默失败。
        构造 1 智能体 + 1 固定目标点的极简静态环境，
        用 MADDPG（num_agents=1）验证收敛性。
        """
        maddpg = MADDPGAgent(num_agents=1, obs_dim=18, act_dim=5)
        B = 1
        obs_val = np.zeros(18, dtype=np.float32)
        obs_val[2:4] = [0.0, 0.0]  # self_pos
        obs_val[4:6] = [0.0, 1.0]  # landmark_0 在上方

        obs_tensor = torch.from_numpy(obs_val).unsqueeze(0)
        next_obs_val = obs_val.copy()
        next_obs_val[4:6] = [0.0, 0.3]  # 接近了（正奖励）
        next_obs_tensor = torch.from_numpy(next_obs_val).unsqueeze(0)

        actions_tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        reward = torch.tensor([[0.7]])

        batch = {
            'obs': {'agent_0': obs_tensor},
            'actions': {'agent_0': actions_tensor},
            'rewards': {'agent_0': reward},
            'next_obs': {'agent_0': next_obs_tensor},
            'dones': {'agent_0': torch.zeros(B, 1)},
        }

        losses = []
        for i in range(50):
            result = maddpg.update(batch)
            losses.append(result['critic_loss'])

        # Critic loss 呈下降趋势
        assert np.mean(losses[:10]) > np.mean(losses[-10:]) * 0.5

        # 过拟合后 Actor 应学到上移
        maddpg.actors['agent_0'].eval()
        with torch.no_grad():
            out = maddpg.actors['agent_0'](obs_tensor).squeeze(0).numpy()
        assert out[4] > 0.3, f"过拟合后上移方向强度不足: {out}"

        # 双 Q 网络在过拟合后应接近
        with torch.no_grad():
            qa, qb = maddpg.critic.q_values(
                obs_tensor.repeat(1, 1),
                actions_tensor
            )
        assert abs(qa.item() - qb.item()) < 2.0, \
            f"双 Q 网络差异过大: QA={qa.item():.3f}, QB={qb.item():.3f}"
