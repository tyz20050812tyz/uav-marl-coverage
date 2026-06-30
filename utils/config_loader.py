"""YAML 配置加载器。

从 configs/default.yaml 加载默认超参数，供实验脚本和训练器使用。
避免各脚本自行重复定义默认值，确保参数来源单一可信。
"""

import os
import yaml
from typing import Any, Dict, Optional

# 缓存已加载的配置
_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载 YAML 配置文件。

    优先使用传入路径，否则自动查找项目根目录下的 configs/default.yaml。

    Args:
        config_path: YAML 配置文件路径（None 则自动查找）

    Returns:
        配置字典
    """
    global _config_cache

    if config_path is not None:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    if _config_cache is not None:
        return _config_cache

    # 自动查找：从当前文件位置向上找到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(project_root, 'configs', 'default.yaml')

    if os.path.exists(default_path):
        with open(default_path, 'r') as f:
            _config_cache = yaml.safe_load(f)
        return _config_cache

    raise FileNotFoundError(
        f"配置文件未找到: {default_path}。"
        f"请确保 configs/default.yaml 存在。"
    )


def get_env_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """获取环境配置子字典。"""
    cfg = config or load_config()
    return cfg.get('env', {})


def get_network_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """获取网络超参数子字典。"""
    cfg = config or load_config()
    return cfg.get('network', {})


def get_training_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """获取训练超参数子字典。"""
    cfg = config or load_config()
    return cfg.get('training', {})


def get_td3_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """获取 TD3 稳定性增强配置子字典。"""
    cfg = config or load_config()
    return cfg.get('td3', {})


def get_rs_maddpg_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """获取 RS-MADDPG 改进模块配置子字典。"""
    cfg = config or load_config()
    return cfg.get('rs_maddpg', {})


def get_experiment_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """获取实验配置子字典。"""
    cfg = config or load_config()
    return cfg.get('experiment', {})
