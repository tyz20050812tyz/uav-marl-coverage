"""Streamlit 前端可复用组件和全局配色常量。"""

import streamlit as st

# ==================== 配色体系 ====================

# 智能体颜色
AGENT_COLORS = {
    0: '#00d2ff',  # 青蓝
    1: '#ff6b6b',  # 珊瑚红
    2: '#ffd93d',  # 琥珀黄
    3: '#6bcb77',  # 薄荷绿
    4: '#c77dff',  # 淡紫
}

# 算法曲线颜色
ALGORITHM_COLORS = {
    'Random': '#888888',
    'IDDPG': '#e76f51',
    'MADDPG': '#2a9d8f',
    'RS-MADDPG': '#457b9d',
    'RSMADDPG': '#457b9d',
}

# 目标点颜色
LANDMARK_COLOR = '#ffffff'
LANDMARK_COVERED = '#00ff88'
LANDMARK_UNCOVERED = '#ffffff40'
COLLISION_FLASH = '#ff4444'


def section_header(title: str, icon: str = ""):
    """统一的分区标题。"""
    if icon:
        st.markdown(f"### {icon} {title}")
    else:
        st.markdown(f"### {title}")


def metric_card(label: str, value, delta=None, help_text: str = ""):
    """统一的指标卡片。"""
    st.metric(label=label, value=value, delta=delta, help=help_text)


def algorithm_label(algo: str) -> str:
    """获取算法中文名称。"""
    labels = {
        'Random': '随机策略 (Random)',
        'IDDPG': '独立 DDPG (IDDPG)',
        'MADDPG': '集中式 MADDPG',
        'RS-MADDPG': 'RS-MADDPG (改进版)',
        'RSMADDPG': 'RS-MADDPG (改进版)',
    }
    return labels.get(algo, algo)


def get_agent_color(idx: int) -> str:
    """获取智能体颜色（循环取色）。"""
    return AGENT_COLORS.get(idx, f'#{idx * 50:02x}{idx * 80:02x}{idx * 110:02x}')


def get_algo_color(algo: str) -> str:
    """获取算法颜色。"""
    return ALGORITHM_COLORS.get(algo, '#cccccc')


# ==================== 全局视觉主题 ====================

GLOBAL_CSS = """
<style>
    /* 深色底板 + 全局字体 */
    .stApp {
        background-color: #0c111d;
        font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', sans-serif;
    }

    /* 主内容区背景 */
    .main .block-container {
        background-color: #0c111d;
        padding-top: 1.5rem;
        max-width: 1380px;
    }

    /* 轻量面板 */
    .stCard, div[data-testid="stMetric"], div[data-testid="stDataFrame"],
    .element-container:has(> div[data-testid="stMetric"]) {
        background-color: #101827;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #20304a;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18);
    }

    /* metric 卡片背景 */
    div[data-testid="stMetric"] {
        background-color: #101827 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }

    div[data-testid="stMetric"] label {
        color: #a8a8b8 !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #20304a;
    }

    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label {
        color: #c0c0d0 !important;
    }

    /* 标题和正文 */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
        letter-spacing: 0 !important;
    }

    p, li, span, div.stMarkdown {
        color: #c9d4e5;
        font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }

    /* 按钮样式 */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        border-color: #2f435f !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #2ec4b6 !important;
        border-color: #2ec4b6 !important;
        color: #07111f !important;
    }

    /* 数据表格 */
    div[data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* 选择器和滑块 */
    div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {
        color: #c0c0d0 !important;
    }

    /* 分割线 */
    hr {
        border-color: #20304a !important;
    }

    /* success/info/warning 消息背景 */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
    }

    /* Plotly 图表容器 */
    .js-plotly-plot, div[data-testid="stPlotlyChart"] {
        background-color: #101827 !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }

    img {
        border-radius: 8px;
        border: 1px solid #20304a;
    }
</style>
"""


def inject_global_css():
    """注入全局深色主题 CSS。应在每个页面的 st.set_page_config 之后调用。"""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
