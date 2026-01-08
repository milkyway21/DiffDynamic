import pandas as pd
import os
from openpyxl import load_workbook

# 读取总体分析.xlsx
try:
    print("正在读取: 总体分析.xlsx")
    df = pd.read_excel('总体分析.xlsx', sheet_name=None)
    print(f"工作表数量: {len(df)}")
    for sheet_name, sheet_df in df.items():
        print(f"\n工作表: {sheet_name}")
        print(f"列名: {list(sheet_df.columns)}")
        print(f"行数: {len(sheet_df)}")
        print(f"前5行数据:")
        print(sheet_df.head())
        print("\n" + "="*80)
except Exception as e:
    print(f"读取总体分析.xlsx时出错: {e}")






