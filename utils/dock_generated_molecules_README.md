# 对接生成分子的工具脚本

## 概述

`dock_generated_molecules.py` 是一个用于对接生成分子的工具脚本。它在sample运行结束后，自动执行以下步骤：

1. 使用 `add_aromatic` 原子编码模式将 `.pt` 文件转换为 `.sdf` 文件
2. 为每个生成分子找到对应的蛋白质口袋
3. 使用 QVina 进行分子对接并计算亲和力

## 使用方法

### 基本用法

```bash
python -m utils.dock_generated_molecules \
  --pt_file result_5.pt \
  --split data/crossdocked_pocket10_pose_split.pt \
  --dataset data/crossdocked_v1.1_rmsd1.0_pocket10 \
  --data_id 5
```

### 参数说明

- `--pt_file`: **必需**，sample 生成的 `.pt` 文件路径
- `--split`: **必需**，split.pt 文件路径
- `--dataset`: **必需**，数据集根目录（包含 index.pkl）
- `--data_id`: **必需**，数据ID（对应 sample 时的 `--data_id` 参数）
- `--output_dir`: 输出目录（默认: `docking_results`）
- `--protein_root`: 蛋白质文件根目录（默认与 dataset 相同）
- `--use_uff`: 是否使用 UFF 优化（默认: True）
- `--size_factor`: 对接盒尺寸因子（默认: 1.2）
- `--debug`: 启用调试模式

## 工作流程

### 1. 文件转换
脚本首先将 `.pt` 文件中的生成分子转换为 RDKit 分子对象：
- 使用 `add_aromatic` 原子编码模式
- 基于原子间距离推断化学键
- 验证分子结构并清理

### 2. 口袋查找
根据提供的 `data_id`，从 split 文件和数据集索引中找到对应的蛋白质口袋文件：
- 读取 split.pt 获取数据集索引
- 从 index.pkl 获取蛋白质口袋相对路径
- 验证文件存在性

### 3. 分子对接
对每个转换成功的分子执行 QVina 对接：
- 准备受体（蛋白质）和配体（分子）文件
- 执行分子对接计算亲和力
- 记录对接结果和化学性质指标

### 4. 结果保存
将所有对接结果保存为 PyTorch 文件，包含：
- 分子结构
- 化学性质指标（QED/SA 等）
- 对接亲和力和 RMSD 值
- 蛋白质口袋信息

## 输出文件

脚本会在输出目录生成以下文件：

- `docked_results_data_{data_id}.pt`: 包含所有对接结果的 PyTorch 文件

## 示例输出

```
Starting conversion of 32 molecules...
Using atom encoding mode: add_aromatic
Successfully converted 30 molecules
Finding pocket for data_id 5...
Found pocket file: /path/to/pocket.pdb
Starting docking of 30 molecules...
Molecule 1/30 docked successfully. Best affinity: -8.234
Molecule 2/30 docked successfully. Best affinity: -7.891
...
Best affinity among all molecules: -9.456
Average affinity: -7.234
Saved 30 docking results to docking_results/docked_results_data_5.pt
```

## 依赖要求

- PyTorch
- RDKit
- NumPy
- EasyDict
- QVina (需要 conda 环境中的 adt 包)

## 注意事项

1. 确保 conda 环境中安装了 `adt` 包用于 QVina 对接
2. 蛋白质口袋文件应该存在于指定路径
3. 对接过程可能需要较长时间，取决于分子数量
4. 生成的 `.pt` 文件应该包含 `pred_ligand_pos` 和 `pred_ligand_v` 键

## 集成到工作流

这个脚本设计用于 sample 脚本运行后的自动对接。通常的使用顺序：

1. 运行 sample：`python scripts/sample_diffusion.py configs/sampling.yml --data_id 5`
2. 对接生成分子：`python -m utils.dock_generated_molecules --pt_file result_5.pt --split ... --dataset ... --data_id 5`


















