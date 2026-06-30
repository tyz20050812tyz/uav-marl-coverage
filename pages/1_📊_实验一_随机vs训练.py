"""实验一：随机策略 vs 训练后策略对比。"""

import streamlit as st
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from frontend.charts import load_logs, plot_training_curves
from frontend.components import section_header, inject_global_css

st.set_page_config(page_title="实验一", page_icon="📊")
inject_global_css()

section_header("实验一：随机策略 vs 训练后策略对比")

st.markdown("""
对比随机策略与训练后的 MADDPG/RS-MADDPG 策略在覆盖率、碰撞次数、完成步数上的差异。
展示智能体从无序移动到协同分工的学习效果。
""")

st.info("此页面从 outputs/logs/ 读取预训练日志。请先运行实验脚本生成日志文件。")

# 尝试加载日志
log_dir = os.path.join(_project_root, 'outputs', 'logs')
algorithms = ['Random', 'MADDPG', 'RS-MADDPG']
logs_data = []

for algo in algorithms:
    # 查找该算法的最新日志
    for f in sorted(os.listdir(log_dir) if os.path.exists(log_dir) else []):
        if f.startswith(algo.lower()) and f.endswith('_logs.json'):
            logs = load_logs(os.path.join(log_dir, f))
            if logs:
                final = logs[-1]
                st.metric(f"{algo}",
                         f"覆盖率 {final['coverage_rate']:.1%} | "
                         f"碰撞 {final['collision_count']:.1f} | "
                         f"奖励 {final['avg_reward']:.1f}")
                logs_data.append((algo, logs))
            break

if logs_data:
    st.divider()
    fig = plot_training_curves(logs_data)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("未找到训练日志。请运行实验脚本。")
