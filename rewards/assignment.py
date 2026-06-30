"""目标分配引导奖励模块。

基于贪心最近分配策略，在每个时间步为每个智能体分配一个软目标点，
引导其向分配目标移动。引入迟滞机制防止目标闪烁（Target Flickering）。
"""

import numpy as np
from typing import Optional


class AssignmentReward:
    """目标分配引导奖励计算器。"""

    def __init__(self, lambda_a: float = 0.5,
                 hysteresis_epsilon: float = 0.05,
                 lock_steps: int = 5):
        """
        初始化分配奖励模块。

        Args:
            lambda_a: 分配奖励权重系数
            hysteresis_epsilon: 迟滞阈值（距离差小于此值不切换分配）
            lock_steps: 前 K 步锁定初始分配
        """
        self.lambda_a = lambda_a
        self.hysteresis_epsilon = hysteresis_epsilon
        self.lock_steps = lock_steps

        # 内部状态
        self._assignment: Optional[dict] = None  # {landmark_idx: agent_idx}
        self._step_counter: int = 0
        self._prev_distances: Optional[np.ndarray] = None  # (N_agents, N_landmarks)

    def reset(self):
        """重置分配状态（每个 episode 开始时调用）。"""
        self._assignment = None
        self._step_counter = 0
        self._prev_distances = None

    def compute(self, agent_positions: np.ndarray,
                landmark_positions: np.ndarray) -> tuple:
        """
        计算目标分配奖励。

        Args:
            agent_positions: (N, 2) 智能体位置
            landmark_positions: (M, 2) 目标点位置 (M == N)

        Returns:
            (reward_dict, assignment_info)
            reward_dict: {agent_name: reward_value}
            assignment_info: 分配信息 dict
        """
        N = agent_positions.shape[0]
        M = landmark_positions.shape[0]

        # 计算距离矩阵
        distances = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                distances[i, j] = np.linalg.norm(
                    agent_positions[i] - landmark_positions[j]
                )

        # 分配逻辑：前 lock_steps 步保持初始分配不变，之后使用迟滞机制
        if self._assignment is None:
            # 首次分配：执行贪心最近分配
            self._assignment = self._greedy_assignment(distances)
            self._prev_distances = distances.copy()
        elif self._step_counter < self.lock_steps:
            # 锁定期内：保持分配不变，不做任何更新
            pass
        else:
            # 锁定期后：使用迟滞机制更新分配，防止目标闪烁
            self._assignment = self._hysteresis_update(distances)

        self._step_counter += 1

        # 计算奖励：R_assign = -lambda_a * d(agent_i, assigned_landmark)
        rewards = {}
        for j, i in self._assignment.items():
            dist = distances[i, j]
            rewards[f'agent_{i}'] = -self.lambda_a * dist

        # 确保所有 agent 都有奖励
        for i in range(N):
            key = f'agent_{i}'
            if key not in rewards:
                rewards[key] = 0.0

        return rewards, {
            'assignment': self._assignment,
            'distances': distances,
        }

    def _greedy_assignment(self, distances: np.ndarray) -> dict:
        """
        贪心最近分配算法。

        对每个目标点，选出距离最近且未被分配给更近目标的智能体。

        Args:
            distances: (N, M) 距离矩阵

        Returns:
            {landmark_idx: agent_idx}
        """
        N, M = distances.shape
        assigned_agents = set()
        assignment = {}

        # 按目标点遍历
        for j in range(M):
            # 按距离排序
            sorted_agents = np.argsort(distances[:, j])
            for i in sorted_agents:
                if i not in assigned_agents:
                    assignment[j] = int(i)
                    assigned_agents.add(i)
                    break

        return assignment

    def _hysteresis_update(self, distances: np.ndarray) -> dict:
        """
        迟滞机制更新分配（冲突安全版）。

        只有当另一个目标点比当前锁定目标点近超过 epsilon 时，才切换分配。
        使用两阶段处理避免分配冲突：
        1. 收集所有切换请求，按距离优势排序
        2. 逐个处理，确保目标点未被其他 agent 占用

        Args:
            distances: (N, M) 当前距离矩阵

        Returns:
            更新后的 {landmark_idx: agent_idx}
        """
        # 阶段一：收集所有切换请求
        switch_requests = []  # [(distance_advantage, agent_i, old_j, new_best_j)]

        for j, i in self._assignment.items():
            current_dist = distances[i, j]
            best_j = j
            best_advantage = 0.0

            for j2 in range(distances.shape[1]):
                if j2 == j:
                    continue
                advantage = current_dist - distances[i, j2]
                if advantage > self.hysteresis_epsilon and advantage > best_advantage:
                    best_advantage = advantage
                    best_j = j2

            if best_j != j:
                switch_requests.append((best_advantage, i, j, best_j))

        # 如果没有切换请求，直接返回
        if not switch_requests:
            return dict(self._assignment)

        # 按距离优势降序排序（优势最大的优先处理）
        switch_requests.sort(key=lambda x: x[0], reverse=True)

        # 阶段二：逐个处理切换，避免冲突
        new_assignment = dict(self._assignment)
        # 追踪在本轮更新中已被占用的目标点
        newly_occupied = set()

        for _, agent_i, old_j, new_best_j in switch_requests:
            # 检查 new_best_j 是否已被占用（包括原始分配和本轮新分配）
            current_owner = new_assignment.get(new_best_j, -1)
            if current_owner == agent_i:
                # 已经是自己的目标点，跳过
                continue

            if current_owner >= 0 and current_owner not in newly_occupied:
                # new_best_j 已分配给其他 agent（原始分配），比较谁更近
                other_agent = current_owner
                my_dist = distances[agent_i, new_best_j]
                other_dist = distances[other_agent, new_best_j]
                if my_dist >= other_dist:
                    # 对方更近或等距，放弃切换
                    continue
                # 我方更近：释放 old_j，占用 new_best_j
                # other_agent 将在后续迭代中处理（如果它也需要切换）
                new_assignment[old_j] = -1
                new_assignment[new_best_j] = agent_i
                newly_occupied.add(new_best_j)
            elif current_owner < 0 or current_owner in newly_occupied:
                # new_best_j 空闲或仅被本轮新占用
                if current_owner in newly_occupied:
                    # 被本轮新占用，比较距离
                    other_agent = new_assignment[new_best_j]
                    my_dist = distances[agent_i, new_best_j]
                    other_dist = distances[other_agent, new_best_j]
                    if my_dist >= other_dist:
                        continue
                new_assignment[old_j] = -1
                new_assignment[new_best_j] = agent_i
                newly_occupied.add(new_best_j)

        # 清理释放的 key
        final_assignment = {}
        for j, i in new_assignment.items():
            if i >= 0:
                final_assignment[j] = i

        return final_assignment
