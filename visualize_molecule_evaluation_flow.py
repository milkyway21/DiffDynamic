#!/usr/bin/env python3
"""
分子评估流程图可视化脚本

生成分子评估流程的流程图，展示从输入到输出的完整流程。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_flowchart():
    """创建分子评估流程图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 28))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 35)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'start': '#4CAF50',      # 绿色 - 开始
        'process': '#2196F3',    # 蓝色 - 处理
        'decision': '#FF9800',   # 橙色 - 判断
        'evaluation': '#9C27B0', # 紫色 - 评估
        'calculation': '#F44336', # 红色 - 计算
        'save': '#00BCD4',       # 青色 - 保存
        'end': '#4CAF50',        # 绿色 - 结束
        'error': '#FF5722'       # 深橙 - 错误处理
    }
    
    # 定义框的样式
    def create_box(x, y, width, height, text, color, style='round'):
        """创建文本框"""
        if style == 'round':
            box = FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0.1", 
                               facecolor=color, 
                               edgecolor='black',
                               linewidth=1.5,
                               alpha=0.8)
        elif style == 'diamond':
            # 创建菱形
            diamond = np.array([[x + width/2, y + height],
                               [x + width, y + height/2],
                               [x + width/2, y],
                               [x, y + height/2]])
            box = mpatches.Polygon(diamond, 
                                 facecolor=color,
                                 edgecolor='black',
                                 linewidth=1.5,
                                 alpha=0.8)
        else:
            box = FancyBboxPatch((x, y), width, height,
                               boxstyle="square,pad=0.1",
                               facecolor=color,
                               edgecolor='black',
                               linewidth=1.5,
                               alpha=0.8)
        
        ax.add_patch(box)
        
        # 添加文本（自动换行）
        words = text.split('\n')
        fontsize = 9 if len(text) > 50 else 10
        for i, word in enumerate(words):
            ax.text(x + width/2, y + height/2 - (i - len(words)/2 + 0.5) * 0.15,
                   word, ha='center', va='center', 
                   fontsize=fontsize, weight='bold', wrap=True)
        
        return box
    
    def create_arrow(x1, y1, x2, y2, style='->', color='black', text=''):
        """创建箭头"""
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle=style,
                              color=color,
                              linewidth=1.5,
                              mutation_scale=20)
        ax.add_patch(arrow)
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.2, text, ha='center', 
                   fontsize=8, style='italic', color=color)
    
    # 标题
    ax.text(5, 34, '分子评估完整流程图', ha='center', va='center',
           fontsize=20, weight='bold', 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    y_pos = 32
    
    # 1. 入口和初始化
    create_box(3, y_pos, 4, 0.8, '1. 入口和初始化\n(main函数)', colors['start'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 2. 加载和验证数据
    create_box(3, y_pos, 4, 0.8, '2. 加载.pt文件\n(load_pt_file)', colors['process'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    create_box(3, y_pos, 4, 0.8, '3. 验证数据\n(validate_pt_data)', colors['process'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 3. 准备评估环境
    create_box(3, y_pos, 4, 0.8, '4. 提取信息\n(坐标、原子类型、文件名)', colors['process'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    create_box(3, y_pos, 4, 0.8, '5. 创建输出目录\n(评估目录、SDF目录、分类目录)', colors['process'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.5
    
    # 4. 分子处理主循环
    create_box(2.5, y_pos, 5, 0.8, '6. 开始分子处理主循环\n(遍历每个分子)', colors['process'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 5. 分子重建
    create_box(3, y_pos, 4, 0.8, '7. 分子重建\n(reconstruct_molecule)', colors['evaluation'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 判断：重建是否成功
    create_box(3, y_pos, 4, 1.2, '重建\n成功?', colors['decision'], style='diamond')
    
    # 失败分支（向右）
    create_arrow(7, y_pos + 0.6, 8.5, y_pos + 0.6, text='否')
    create_box(8.5, y_pos, 1.2, 1.2, '跳过\n记录失败', colors['error'])
    create_arrow(9.7, y_pos + 0.6, 9.7, y_pos - 2)
    
    # 成功分支（向下）
    create_arrow(5, y_pos, 5, y_pos - 1.3, text='是')
    y_pos -= 1.5
    
    # 6. 分子评估（使用子进程隔离）
    create_box(2, y_pos, 6, 0.8, '8. 分子评估\n(evaluate_single_molecule_isolated)', colors['evaluation'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 评估子步骤（左侧详细流程）
    eval_y = y_pos
    eval_steps = [
        ('8.1 RDKit验证', colors['evaluation']),
        ('8.2 化学指标\n(QED, SA, logP, Lipinski)', colors['evaluation']),
        ('8.3 TPSA计算', colors['evaluation']),
        ('8.4 基础结构信息\n(原子数、键数、环数、分子量)', colors['evaluation']),
        ('8.5 PAINS检测', colors['evaluation']),
        ('8.6 稳定性评估\n(check_stability)', colors['evaluation']),
        ('8.7 Tanimoto相似度', colors['evaluation']),
        ('8.8 RDKit RMSD', colors['evaluation']),
        ('8.9 构象能量', colors['evaluation']),
        ('8.10 Lilly Medchem Rules', colors['evaluation']),
        ('8.11 生成SMILES', colors['evaluation']),
        ('8.12 结构指标\n(原子类型、键长、距离JSD)', colors['evaluation']),
        ('8.13 AutoDock Vina对接\n(Dock, Score Only, Minimize)', colors['evaluation']),
    ]
    
    # 在左侧绘制评估子步骤
    left_x = 0.5
    left_y = eval_y
    for i, (text, color) in enumerate(eval_steps):
        create_box(left_x, left_y, 1.3, 0.5, text, color, style='square')
        if i < len(eval_steps) - 1:
            create_arrow(left_x + 0.65, left_y, left_x + 0.65, left_y - 0.6)
        left_y -= 0.6
    
    # 主流程继续（右侧）
    create_box(3, y_pos, 4, 0.8, '8. 评估结果汇总', colors['evaluation'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 7. 综合评分计算
    create_box(2.5, y_pos, 5, 0.8, '9. 计算综合评分\n(calculate_comprehensive_score)', colors['calculation'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 评分计算子步骤（右侧）
    score_steps = [
        ('9.1 Vina亲和力归一化', colors['calculation']),
        ('9.2 基础分加权和\n(40%亲和力+30%QED+20%SA+10%Lipinski)', colors['calculation']),
        ('9.3 惩罚系数\n(PAINS×稳定性)', colors['calculation']),
        ('9.4 最终得分\n(100×基础分×惩罚)', colors['calculation']),
    ]
    
    right_x = 8
    right_y = y_pos + 0.5
    for i, (text, color) in enumerate(score_steps):
        create_box(right_x, right_y, 1.5, 0.5, text, color, style='square')
        if i < len(score_steps) - 1:
            create_arrow(right_x + 0.75, right_y, right_x + 0.75, right_y - 0.6)
        right_y -= 0.6
    
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 8. 生成分子身份证
    create_box(3, y_pos, 4, 0.8, '10. 生成分子身份证\n(蛋白质ID+时间+评分)', colors['process'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 9. 保存结果
    create_box(2.5, y_pos, 5, 0.8, '11. 保存SDF文件\n(添加所有属性)', colors['save'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 判断：评分分类
    create_box(3, y_pos, 4, 1.2, '评分\n分类?', colors['decision'], style='diamond')
    
    # 分类分支（向右）
    create_arrow(7, y_pos + 0.6, 8.5, y_pos + 0.6, text='是')
    create_box(8.5, y_pos, 1.2, 1.2, '复制到\n分类目录\n(65-70/70-80/80+)', colors['save'])
    create_arrow(9.7, y_pos + 0.6, 9.7, y_pos - 1.5)
    
    # 继续主流程
    create_arrow(5, y_pos, 5, y_pos - 1.3)
    y_pos -= 1.5
    
    # 保存到结果列表
    create_box(3, y_pos, 4, 0.8, '12. 保存到结果列表', colors['save'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 判断：是否每16个分子
    create_box(3, y_pos, 4, 1.2, '每16个\n分子?', colors['decision'], style='diamond')
    
    # 是分支（向右）
    create_arrow(7, y_pos + 0.6, 8.5, y_pos + 0.6, text='是')
    create_box(8.5, y_pos, 1.2, 1.2, '保存\n中间结果', colors['save'])
    create_arrow(9.7, y_pos + 0.6, 9.7, y_pos - 1.5)
    
    # 否分支（继续循环）
    create_arrow(5, y_pos, 5, y_pos - 1.3, text='否')
    y_pos -= 1.5
    
    # 判断：是否还有分子
    create_box(3, y_pos, 4, 1.2, '还有\n分子?', colors['decision'], style='diamond')
    
    # 是分支（向上循环）
    create_arrow(2, y_pos + 0.6, 2, y_pos + 3, text='是')
    create_arrow(2, y_pos + 3, 5, y_pos + 3)
    create_arrow(5, y_pos + 3, 5, 30.8)
    
    # 否分支（继续）
    create_arrow(5, y_pos, 5, y_pos - 1.3, text='否')
    y_pos -= 1.5
    
    # 最终统计和保存
    create_box(2, y_pos, 6, 0.8, '13. 最终统计和保存\n(Excel、JSON、失败信息)', colors['save'])
    create_arrow(5, y_pos, 5, y_pos - 0.9)
    y_pos -= 1.2
    
    # 结束
    create_box(3, y_pos, 4, 0.8, '14. 评估完成', colors['end'])
    
    # 添加图例
    legend_x = 0.2
    legend_y = 2
    legend_items = [
        ('开始/结束', colors['start']),
        ('处理步骤', colors['process']),
        ('评估步骤', colors['evaluation']),
        ('计算步骤', colors['calculation']),
        ('保存步骤', colors['save']),
        ('判断节点', colors['decision']),
        ('错误处理', colors['error']),
    ]
    
    ax.text(legend_x, legend_y + 1, '图例', fontsize=12, weight='bold')
    for i, (label, color) in enumerate(legend_items):
        y = legend_y - i * 0.4
        box = FancyBboxPatch((legend_x, y), 0.3, 0.25,
                           boxstyle="round,pad=0.05",
                           facecolor=color,
                           edgecolor='black',
                           linewidth=1)
        ax.add_patch(box)
        ax.text(legend_x + 0.4, y + 0.125, label, fontsize=9, va='center')
    
    # 添加说明
    ax.text(5, 0.5, 
           '注：评估步骤(8.1-8.13)在左侧详细展示，评分计算(9.1-9.4)在右侧详细展示',
           ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    return fig

def save_flowchart(output_path='molecule_evaluation_flowchart.png', dpi=300):
    """保存流程图"""
    fig = create_flowchart()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"流程图已保存到: {output_path}")
    plt.close(fig)

def show_flowchart():
    """显示流程图"""
    fig = create_flowchart()
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    import sys
    import os
    
    # 默认保存到当前目录
    output_file = 'molecule_evaluation_flowchart.png'
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    # 支持多种输出格式
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    output_name = os.path.splitext(os.path.basename(output_file))[0]
    
    print("正在生成分子评估流程图...")
    
    # 生成PNG格式（高分辨率）
    png_file = os.path.join(output_dir, f'{output_name}.png')
    save_flowchart(png_file, dpi=300)
    
    # 生成PDF格式（矢量图，可缩放）
    try:
        pdf_file = os.path.join(output_dir, f'{output_name}.pdf')
        fig = create_flowchart()
        fig.savefig(pdf_file, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"PDF格式已保存到: {pdf_file}")
    except Exception as e:
        print(f"PDF格式保存失败: {e}")
    
    print("完成!")
    print(f"\n生成的文件:")
    print(f"  - PNG: {png_file}")
    if 'pdf_file' in locals():
        print(f"  - PDF: {pdf_file}")
    
    # 如果需要在交互式环境中显示，取消下面的注释
    # show_flowchart()

