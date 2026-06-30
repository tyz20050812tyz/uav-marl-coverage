"""原始环境奖励提取模块。

从 PettingZoo 环境返回的 env_rewards dict 中提取覆盖奖励，
提供与塑形奖励模块一致的接口，方便统一组合和未来扩展。
"""

from typing import Dict


class BaseEnvReward:
    """原始环境奖励提取器。

    封装 env_rewards 的读取，保持与 AssignmentReward/RedundancyPenalty/
    SafetyPenalty 一致的 compute() 接口，使奖励塑形管道可统一遍历。
    """

    def __init__(self):
        """初始化原始奖励提取器（无状态）。"""
        pass

    def reset(self):
        """重置状态（BaseEnvReward 无内部状态，空操作）。"""
        pass

    def compute(self, env_rewards: Dict[str, float]) -> Dict[str, float]:
        """
        提取原始环境奖励。

        Args:
            env_rewards: PettingZoo 环境返回的 {agent_name: reward}

        Returns:
            与输入一致的奖励 dict
        """
        return dict(env_rewards)
