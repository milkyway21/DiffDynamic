#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
USP7抑制剂数据获取脚本
功能：从ChEMBL数据库获取USP7抑制剂的活性数据，并进行筛选和整理（带实时进度显示）
"""

import csv
import sys
from datetime import datetime
import traceback
from math import ceil

# 检查并安装所需依赖
try:
    from chembl_webresource_client.new_client import new_client
    import pandas as pd
    from tqdm import tqdm  # 进度条库
except ImportError:
    print("正在安装必要的依赖包...")
    try:
        import subprocess
        # 静默安装，避免输出刷屏
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "chembl-webresource-client", "pandas", "tqdm"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        from chembl_webresource_client.new_client import new_client
        import pandas as pd
        from tqdm import tqdm
        print("依赖包安装完成！")
    except subprocess.CalledProcessError as e:
        print(f"依赖包安装失败：{e}")
        sys.exit(1)

def unit_conversion(df):
    """
    单位标准化：统一转换为μM（微摩尔）
    ChEMBL中常见单位：nM（纳摩尔）、μM（微摩尔）、mM（毫摩尔）
    转换关系：1 μM = 1000 nM，1 mM = 1000 μM
    """
    print("正在进行单位标准化转换...")
    # 先确保单位列是字符串类型
    df["单位"] = df["单位"].astype(str).str.strip()
    df["标准活性值"] = pd.to_numeric(df["标准活性值"], errors="coerce")

    # 定义转换规则
    conversion_rules = {
        "nM": 0.001,    # nM转μM：除以1000
        "μM": 1.0,      # 已为μM，不转换
        "mM": 1000.0,   # mM转μM：乘以1000
        "mol/L": 1000000.0  # mol/L转μM：乘以1e6（极少出现）
    }

    # 遍历转换规则，更新活性值和单位
    for unit, factor in conversion_rules.items():
        try:
            mask = df["单位"] == unit
            # 使用where避免无效值相乘
            df.loc[mask, "标准活性值"] = df.loc[mask, "标准活性值"].where(pd.notna(df.loc[mask, "标准活性值"]), None) * factor
            df.loc[mask, "单位"] = "μM"
        except Exception as e:
            print(f"转换单位 {unit} 时出错: {e}")
            continue

    # 剔除无法识别的单位
    unknown_units = df[df["单位"] != "μM"]["单位"].unique()
    if len(unknown_units) > 0:
        print(f"警告：发现未识别的单位 {unknown_units}，已剔除对应数据")
        df = df[df["单位"] == "μM"]
    
    # 剔除转换后无效的活性值
    df = df.dropna(subset=["标准活性值"])

    return df

def get_usp7_inhibitors():
    """
    从ChEMBL数据库获取USP7抑制剂数据（带实时进度显示）
    返回筛选后的DataFrame
    """
    print("开始从ChEMBL数据库获取USP7抑制剂数据...")
    
    # 设置USP7靶点ID（CHEMBL240，USP7别名HAUSP）
    target_id = "CHEMBL240"
    
    # 初始化ChEMBL客户端
    activities = new_client.activity

    try:
        # 第一步：先查询符合条件的总数据量（用于进度条）
        print(f"正在查询靶点 {target_id} (USP7) 的数据总量...")
        # 使用列表长度替代count()方法
        activity_data = list(activities.filter(
            target_chembl_id=target_id,
            standard_value__isnull=False,
            standard_units__isnull=False,
            molecule_type="Small molecule",
            standard_type__in=["IC50", "Ki", "EC50"]
        ).only(
            "molecule_chembl_id", "canonical_smiles", "standard_value", "standard_units", "standard_type"
        ))
        total_count = len(activity_data)
        print(f"共检测到 {total_count} 条原始活性数据，开始分批加载...")

        if total_count == 0:
            print("未查询到任何USP7抑制剂活性数据")
            return pd.DataFrame()

        # 第二步：直接处理已获取的所有数据
        all_results = []  # 存储所有结果的列表

        # 创建进度条
        pbar = tqdm(total=total_count, desc="加载ChEMBL数据", unit="条")

        # 直接使用已获取的activity_data
        for item in activity_data:
            all_results.append(item)
            pbar.update(1)

        pbar.close()  # 关闭进度条
        print("原始数据加载完成，开始处理筛选...")

        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 数据预处理和筛选
        # 重命名列以匹配需求
        df = df.rename(columns={
            "molecule_chembl_id": "ChEMBL编号",
            "canonical_smiles": "Smiles字符串",
            "standard_type": "活性类型",
            "standard_value": "标准活性值",
            "standard_units": "单位"
        })
        
        # 转换活性值为数值类型并剔除无效值
        df["标准活性值"] = pd.to_numeric(df["标准活性值"], errors="coerce")
        df = df.dropna(subset=["标准活性值", "Smiles字符串"])

        # 单位标准化（统一为μM）
        df = unit_conversion(df)

        # 筛选活性值在合理范围内的条目 (0 < 活性值 < 1000 μM，排除异常值)
        valid_mask = (df["标准活性值"] > 0) & (df["标准活性值"] < 1000)
        df = df[valid_mask].copy()

        # 按活性值升序排序（值越小活性越强）
        df = df.sort_values(by="标准活性值", ascending=True)
        
        # 保留唯一的ChEMBL ID（每个化合物只保留活性最强的条目）
        df = df.drop_duplicates(subset=["ChEMBL编号"], keep="first").reset_index(drop=True)

    except Exception as e:
        print(f"数据查询/处理失败：{e}")
        traceback.print_exc()
        return pd.DataFrame()

    print(f"共筛选到 {len(df)} 个有效USP7抑制剂")
    return df

def save_to_csv(df, filename="USP7_inhibitors_ChEMBL.csv"):
    """
    将筛选后的数据保存为CSV文件（UTF-8编码，兼容Excel）
    """
    if df.empty:
        print("没有有效数据可保存")
        return False
    
    try:
        # index=False：不保存行索引；quoting=csv.QUOTE_ALL：所有字段加引号，避免Smiles特殊字符解析错误
        df.to_csv(
            filename, 
            index=False, 
            encoding="utf-8-sig",  # utf-8-sig兼容Excel打开中文乱码
            quoting=csv.QUOTE_ALL
        )
        print(f"数据已保存至: {filename}")
        return True
    except Exception as e:
        print(f"保存CSV文件失败: {str(e)}")
        return False

def generate_markdown_report(df, filename="USP7_inhibitors_report.md"):
    """
    生成包含数据的Markdown报告，适配群聊/文档展示
    """
    if df.empty:
        print("没有有效数据可生成报告")
        return False
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            # 报告标题和基础信息
            f.write("# USP7抑制剂活性数据汇总（ChEMBL来源）\n\n")
            f.write(f"*数据生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
            f.write(f"*USP7靶点ChEMBL ID: CHEMBL240 (别名HAUSP)*\n\n")
            
            # 数据概览
            f.write("## 数据概览\n")
            f.write(f"- 共整理 **{len(df)}** 个高置信度USP7小分子抑制剂\n")
            f.write("- 所有化合物均包含完整的ChEMBL编号、Smiles结构、活性类型及标准化活性值\n")
            f.write("- 活性值已统一转换为 **μM（微摩尔）**，值越小代表抑制活性越强\n\n")
            
            # 数据表格
            f.write("## 详细数据表格\n")
            f.write("| ChEMBL编号 | Smiles字符串 | 活性类型 | 标准活性值（μM） | 单位 |\n")
            f.write("|------------|--------------|----------|------------------|------|\n")
            
            # 遍历数据写入表格（处理长Smiles字符串的换行问题）
            for _, row in df.iterrows():
                # 替换Smiles中的特殊字符（避免Markdown解析错误）
                smiles = row["Smiles字符串"].replace("|", "\\|").replace("\n", "")
                f.write(
                    f"| {row['ChEMBL编号']} | {smiles} | {row['活性类型']} | {row['标准活性值']:.4f} | {row['单位']} |\n"
                )
            
            # 筛选条件说明
            f.write("\n## 数据筛选规则\n")
            f.write("1. 仅保留**小分子化合物**（排除多肽、聚合物、混合物）；\n")
            f.write("2. 活性类型优先保留IC50（半数抑制浓度）、Ki（抑制常数）、EC50（半数有效浓度）；\n")
            f.write("3. 剔除Smiles为空、活性值异常（≤0或≥1000 μM）的条目；\n")
            f.write("4. 每个ChEMBL编号仅保留**活性最强**的一条数据；\n")
            f.write("5. 活性单位统一转换为μM，确保数据可横向对比。\n\n")
            
            # 统计信息
            f.write("## 活性统计\n")
            activity_type_count = df["活性类型"].value_counts().to_dict()
            f.write("- 活性类型分布：\n")
            for act_type, count in activity_type_count.items():
                f.write(f"  - {act_type}：{count} 个化合物\n")
            f.write(f"- 最强活性化合物：{df.iloc[0]['ChEMBL编号']}（{df.iloc[0]['标准活性值']:.4f} μM）\n")

        print(f"Markdown报告已生成: {filename}")
        return True
    except Exception as e:
        print(f"生成Markdown报告失败: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """
    主函数：串联数据获取、处理、保存、报告生成流程
    """
    print("="*50)
    print("USP7抑制剂数据获取脚本启动（带实时进度）")
    print("="*50)
    try:
        # 获取并筛选数据
        df = get_usp7_inhibitors()
        
        if not df.empty:
            # 保存为CSV文件
            save_to_csv(df)
            
            # 生成Markdown报告（适合发群聊）
            generate_markdown_report(df)
            
            # 控制台输出预览（前5条）
            print("\n" + "="*50)
            print("数据预览（前5条，活性由强到弱）:")
            print("="*50)
            print(df.head().to_string(index=False))
            
            # 输出关键统计信息
            print("\n" + "="*50)
            print("统计信息:")
            print("="*50)
            print(f"- 总有效化合物数量: {len(df)}")
            print(f"- 活性类型分布: {df['活性类型'].value_counts().to_dict()}")
            print(f"- 最强活性化合物: {df.iloc[0]['ChEMBL编号']}")
            print(f"- 最强活性值: {df.iloc[0]['标准活性值']:.4f} {df.iloc[0]['单位']}")
        else:
            print("未获取到有效数据，脚本执行结束")

    except Exception as e:
        print(f"脚本执行异常: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()