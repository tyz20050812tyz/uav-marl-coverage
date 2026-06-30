"""2D 仿真区域实时渲染模块。

在 PettingZoo 原始帧上叠加覆盖圈、安全圈、拖尾轨迹等可视化元素。
"""

import numpy as np
import cv2


def render_frame(env, world_state: dict,
                 coverage_radius: float = 0.24,
                 safe_distance: float = 0.2,
                 trail_length: int = 10,
                 trail_history: list = None) -> np.ndarray:
    """
    在环境原始帧上叠加可视化元素。

    Args:
        env: SimpleSpreadWrapper 实例
        world_state: get_world_state() 返回的字典
        coverage_radius: 覆盖圈半径
        safe_distance: 安全圈半径
        trail_length: 拖尾历史长度
        trail_history: 过去 N 帧的智能体位置列表

    Returns:
        (H, W, 3) RGB 帧数组
    """
    frame = env.render()
    if frame is None:
        return np.zeros((700, 700, 3), dtype=np.uint8)

    H, W = frame.shape[:2]
    # 渲染尺寸下的坐标映射
    # MPE 世界坐标为 [-1, 1]，渲染为 [0, W] × [0, H]
    scale_x = W / 2.0
    scale_y = H / 2.0
    offset_x = W / 2.0
    offset_y = H / 2.0

    def world_to_pixel(pos):
        """将世界坐标 [-1, 1] 映射到像素坐标。"""
        x = int(pos[0] * scale_x + offset_x)
        y = int(offset_y - pos[1] * scale_y)  # MPE y 轴方向
        return np.clip(x, 0, W - 1), np.clip(y, 0, H - 1)

    agent_positions = world_state['agent_positions']
    landmark_positions = world_state['landmark_positions']

    # 叠加安全圈（半透明蓝色，在智能体周围）
    for i, pos in enumerate(agent_positions):
        cx, cy = world_to_pixel(pos)
        radius_px = int(safe_distance * scale_x)
        if radius_px > 0:
            cv2.circle(frame, (cx, cy), radius_px, (100, 180, 255), 1, cv2.LINE_AA)

    # 叠加覆盖圈（半透明绿色，在目标点周围）
    for j, pos in enumerate(landmark_positions):
        cx, cy = world_to_pixel(pos)
        radius_px = int(coverage_radius * scale_x)
        if radius_px > 0:
            cv2.circle(frame, (cx, cy), radius_px, (0, 200, 100), 1, cv2.LINE_AA)

    # 叠加拖尾轨迹
    if trail_history and len(trail_history) >= 2:
        trail = trail_history[-trail_length:]
        N = len(trail[0]) if trail else 0
        for i in range(N):
            pts = []
            for t in trail:
                px, py = world_to_pixel(t[i])
                pts.append((px, py))
            if len(pts) >= 2:
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    color = get_trail_color(i, alpha)
                    cv2.line(frame, pts[k - 1], pts[k], color, 1, cv2.LINE_AA)

    return frame


def get_trail_color(agent_idx: int, alpha: float):
    """根据智能体索引和透明度计算拖尾颜色（BGR for OpenCV）。"""
    from frontend.components import get_agent_color
    hex_color = get_agent_color(agent_idx)
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    # OpenCV 使用 BGR
    return (
        int(b * alpha),
        int(g * alpha),
        int(r * alpha),
    )
