"""智能体基类。提供 save/load 接口。"""

import torch
import os


class BaseAgent:
    """所有智能体的基类。"""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    def act(self, obs, **kwargs):
        """根据观测选择动作。"""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """执行一次学习更新。"""
        raise NotImplementedError

    def save(self, path: str):
        """保存模型。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def load(self, path: str):
        """加载模型。"""
        pass

    def _save_state_dict(self, obj, path: str):
        """保存 state_dict。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(obj.state_dict(), path)

    def _load_state_dict(self, obj, path: str):
        """加载 state_dict。"""
        obj.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
