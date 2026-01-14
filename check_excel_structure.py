#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

excel_file = r'F:\DiffDynamic\DiffDynamic\20250113\batch_evaluation_summary_20260108_001301_gfquadratic_1.0_0.0_tl750_lslambda_80p0_20p0_lsstep_1p0_lsnoise_0p0_rflambda_20p0_2p0_rfstep_0p2_rfnoise_0p08.xlsx'

# 读取所有sheet
xl_file = pd.ExcelFile(excel_file, engine='openpyxl')
print("Sheet names:", xl_file.sheet_names)

# 读取第一个sheet（详细数据）
df = pd.read_excel(excel_file, sheet_name=0, engine='openpyxl')
print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# 查找包含Vina的列
vina_cols = [col for col in df.columns if 'Vina' in str(col)]
print("\nVina-related columns:")
for col in vina_cols:
    print(f"  {col}")
    print(f"    Sample values: {df[col].head(5).tolist()}")

# 读取统计信息sheet
try:
    df_stats = pd.read_excel(excel_file, sheet_name='统计信息', engine='openpyxl')
    print("\n统计信息 sheet:")
    print(df_stats)
except:
    print("\n无法读取统计信息sheet")

