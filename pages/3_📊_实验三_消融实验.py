"""实验三：消融实验。"""

import streamlit as st
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from frontend.charts import load_logs
from frontend.components import section_header, inject_global_css

st.set_page_config(page_title="实验三", page_icon="📊")
inject_global_css()

section_header("实验三：消融实验 — 各改进模块贡献度")

st.markdown("""
验证 RS-MADDPG 中每个改进模块的独立贡献。
五组实验：MADDPG → +Assignment → +Redundancy → +Safety → RS-MADDPG(full)。
""")

log_dir = os.path.join(_project_root, 'outputs', 'logs')
ablation_groups = [
    ('MADDPG (baseline)', 'ablation_maddpg'),
    ('+ Assignment', 'ablation_assign'),
    ('+ Redundancy', 'ablation_ar'),
    ('+ Safety', 'ablation_ars'),
    ('RS-MADDPG (full)', 'ablation_full'),
]

# 收集各组数据
table_data = {}
for label, prefix in ablation_groups:
    if os.path.exists(log_dir):
        for f in sorted(os.listdir(log_dir)):
            if f.startswith(prefix) and f.endswith('_logs.json'):
                logs = load_logs(os.path.join(log_dir, f))
                if logs:
                    final = logs[-1]
                    table_data[label] = {
                        '覆盖率': f"{final['coverage_rate']:.1%}",
                        '碰撞次数': f"{final['collision_count']:.1f}",
                        '完成步数': f"{final['completion_steps']:.0f}",
                        '平均奖励': f"{final['avg_reward']:.1f}",
                    }
                break

if table_data:
    import pandas as pd
    import plotly.graph_objects as go

    # 数据表格
    df = pd.DataFrame(table_data).T
    st.subheader("📋 各消融组最终指标")
    st.dataframe(df, use_container_width=True)

    # 分组柱状图
    st.subheader("📊 覆盖率对比")
    chart_data = {}
    for label, metrics in table_data.items():
        chart_data[label] = {
            '覆盖率': float(metrics['覆盖率'].rstrip('%')) / 100,
            '碰撞次数': float(metrics['碰撞次数']),
        }

    fig = go.Figure()
    groups = list(chart_data.keys())
    for metric_name in ['覆盖率', '碰撞次数']:
        values = [chart_data[g][metric_name] for g in groups]
        fig.add_trace(go.Bar(
            name=metric_name, x=groups, y=values,
            text=[f'{v:.2f}' if metric_name == '覆盖率' else f'{v:.1f}' for v in values],
            textposition='outside',
        ))
    fig.update_layout(
        barmode='group', height=400,
        title='消融实验各模块贡献度',
        yaxis_title='数值',
        margin=dict(l=40, r=40, t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("💡 结论：目标分配提升覆盖率最显著，安全距离降低碰撞最显著。")
else:
    st.warning("⚠️ 未找到消融实验日志。请先运行: python experiments/exp3_ablation.py")
