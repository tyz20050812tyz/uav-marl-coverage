"""simple_spread_v3 环境封装模块。

封装 PettingZoo MPE simple_spread_v3 环境，提供统一接口：
- reset / step / render
- 便捷属性（obs_dim, act_dim, num_agents, world_size）
- 支持可变智能体数量（N=3/4/5）
- 支持 Supersuit 向量化
"""

import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import numpy as np
from typing import Dict, Optional, Tuple, List

from pettingzoo.mpe import simple_spread_v3
import supersuit as ss


class SimpleSpreadWrapper:
    """封装 simple_spread_v3 环境，统一接口。"""

    def __init__(self, num_agents: int = 3, local_ratio: float = 0.5,
                 max_cycles: int = 50, continuous_actions: bool = True,
                 render_mode: str = 'rgb_array'):
        """
        初始化环境封装。

        Args:
            num_agents: 智能体（目标点）数量
            local_ratio: 局部奖励与全局奖励的权重比例
            max_cycles: 每 episode 最大步数
            continuous_actions: 是否使用连续动作空间
            render_mode: 渲染模式
        """
        self.num_agents = num_agents
        self.local_ratio = local_ratio
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode

        self._env = None
        self._agent_names: List[str] = []
        self._build_env()

    def _build_env(self):
        """构建底层 PettingZoo 环境。"""
        self._env = simple_spread_v3.parallel_env(
            N=self.num_agents,
            local_ratio=self.local_ratio,
            max_cycles=self.max_cycles,
            continuous_actions=self.continuous_actions,
            render_mode=self.render_mode,
        )
        self._agent_names = self._env.possible_agents

    def set_config(self, num_agents: Optional[int] = None,
                   local_ratio: Optional[float] = None,
                   max_cycles: Optional[int] = None):
        """
        动态调整环境配置（重建环境）。

        Args:
            num_agents: 新的智能体数量
            local_ratio: 新的 local_ratio
            max_cycles: 新的 max_cycles
        """
        if num_agents is not None:
            self.num_agents = num_agents
        if local_ratio is not None:
            self.local_ratio = local_ratio
        if max_cycles is not None:
            self.max_cycles = max_cycles
        self.close()
        self._build_env()

    @property
    def agent_names(self) -> List[str]:
        """返回智能体名称列表。"""
        return self._agent_names

    @property
    def obs_dim(self) -> int:
        """返回每个智能体的观测维度。"""
        return self._env.observation_space(self._agent_names[0]).shape[0]

    @property
    def act_dim(self) -> int:
        """返回每个智能体的动作维度。"""
        return self._env.action_space(self._agent_names[0]).shape[0]

    @property
    def world_size(self) -> float:
        """返回环境世界边长（用于计算覆盖半径、安全距离）。"""
        return 2.0  # MPE 默认世界范围为 [-1, 1] × [-1, 1]

    # --- 便捷方法别名（兼容计划书接口规范）---

    def get_obs_dim(self) -> int:
        """便捷方法：返回观测维度（与 obs_dim 属性等价）。"""
        return self.obs_dim

    def get_act_dim(self) -> int:
        """便捷方法：返回动作维度（与 act_dim 属性等价）。"""
        return self.act_dim

    def get_num_agents(self) -> int:
        """便捷方法：返回智能体数量（与 num_agents 属性等价）。"""
        return self.num_agents

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """重置环境，返回初始观测和信息。"""
        obs, info = self._env.reset()
        return obs, info

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
            Dict[str, np.ndarray], Dict[str, float],
            Dict[str, bool], Dict[str, bool], Dict]:
        """
        执行一步动作。

        Args:
            actions: {agent_name: action_array} 格式的动作字典

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        obs, rewards, terminations, truncations, infos = self._env.step(actions)
        return obs, rewards, terminations, truncations, infos

    def render(self) -> np.ndarray:
        """渲染当前帧，返回 (H, W, 3) RGB 数组。"""
        frame = self._env.render()
        if frame is None:
            # 如果 render_mode 不是 rgb_array，返回空数组
            return np.zeros((700, 700, 3), dtype=np.uint8)
        return frame

    def close(self):
        """关闭环境。"""
        if self._env is not None:
            self._env.close()

    def get_world_state(self) -> Dict[str, np.ndarray]:
        """
        获取当前世界状态（用于奖励塑形模块）。

        Returns:
            dict with:
                'agent_positions': (N, 2) 智能体位置
                'landmark_positions': (N, 2) 目标点位置
                'agent_velocities': (N, 2) 智能体速度
        """
        world = self._env.unwrapped.world
        agent_positions = np.array([a.state.p_pos for a in world.agents])
        agent_velocities = np.array([a.state.p_vel for a in world.agents])
        landmark_positions = np.array([lm.state.p_pos for lm in world.landmarks])
        return {
            'agent_positions': agent_positions,
            'landmark_positions': landmark_positions,
            'agent_velocities': agent_velocities,
        }

    def make_vec_env(self, num_envs: int = 4) -> 'SimpleSpreadWrapper':
        """
        创建向量化环境（使用 Supersuit）。

        .. warning::
            此方法会原地替换 self._env 为向量化版本，之后 render()、
            get_world_state() 等方法将无法正常工作。
            建议在使用前创建环境副本，或仅将此方法用于批量训练。

        Args:
            num_envs: 并行环境数量

        Returns:
            self（原地修改，将 _env 替换为向量化版本）
        """
        import warnings
        warnings.warn(
            "make_vec_env() 将替换 self._env 为向量化版本，"
            "后续 render()/get_world_state() 将不可用。"
            "建议在调用前通过 SimpleSpreadWrapper(...) 创建独立副本。",
            UserWarning, stacklevel=2,
        )
        # 先关闭当前环境
        self.close()
        # 创建多个独立环境并打包
        self._env = simple_spread_v3.parallel_env(
            N=self.num_agents,
            local_ratio=self.local_ratio,
            max_cycles=self.max_cycles,
            continuous_actions=self.continuous_actions,
            render_mode=self.render_mode,
        )
        self._env = ss.pettingzoo_env_to_vec_env_v1(self._env)
        self._env = ss.concat_vec_envs_v1(
            self._env, num_vec_envs=num_envs, num_cpus=min(num_envs, 4),
            base_class='gymnasium',
        )
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
