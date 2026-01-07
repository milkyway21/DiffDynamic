# 采样策略配置文件说明

本目录包含多个不同的跳步采样策略配置文件，您可以根据需要选择合适的策略。

## 策略概览

### 1. sampling_conservative.yml - 保守精细型
- **time_boundary**: 750
- **总采样步数**: 约90步
- **实际长度**: 约24.35（大步阶段步数×0.45 + 精炼阶段步数×0.25）
- **特点**: 
  - 较小的跳步，更细致的采样
  - 适合需要高质量结果的场景
  - 计算时间较长但结果更稳定

### 2. sampling_balanced.yml - 平衡型
- **time_boundary**: 600
- **总采样步数**: 约70步
- **实际长度**: 约19.95（大步阶段步数×0.45 + 精炼阶段步数×0.25）
- **特点**:
  - 平衡效率和质量的策略
  - 推荐作为默认策略使用
  - 适合大多数应用场景

### 3. sampling_aggressive.yml - 激进快速型
- **time_boundary**: 500
- **总采样步数**: 约55步
- **实际长度**: 约16.4（大步阶段步数×0.45 + 精炼阶段步数×0.25）
- **特点**:
  - 大跳步，快速采样
  - 适合快速探索或计算资源有限的情况
  - 速度最快但可能牺牲一些精度

### 4. sampling_moderate.yml - 中等保守型
- **time_boundary**: 700
- **总采样步数**: 约80步
- **实际长度**: 约21.85（大步阶段步数×0.45 + 精炼阶段步数×0.25）
- **特点**:
  - 介于保守和平衡之间
  - 提供较好的质量保证
  - 适合对质量有一定要求但不想太慢的场景

## 使用方法

要切换策略，只需将选定的配置文件复制到主配置目录：

```bash
# 使用平衡策略
cp configs/sampling_strategies/sampling_balanced.yml configs/sampling.yml

# 使用保守策略
cp configs/sampling_strategies/sampling_conservative.yml configs/sampling.yml

# 使用激进策略
cp configs/sampling_strategies/sampling_aggressive.yml configs/sampling.yml

# 使用中等保守策略
cp configs/sampling_strategies/sampling_moderate.yml configs/sampling.yml
```

## 参数说明

所有策略都保持以下参数不变：
- `LSstepsize`: 0.45 (黄金比例步长)
- `RFstepsize`: 0.25 (配合LS步长的RF步长)
- `RFnoise`: 0.05 (经验最优噪声)
- `LSnoise`: 0 (无变化)

主要调整的参数：
- `time_boundary`: 控制精炼阶段的起始点（对于 num_steps=1000，推荐范围：300-800）
  - 保守策略：750（更长精炼阶段）
  - 中等保守：700
  - 平衡策略：600
  - 激进策略：500（更短精炼阶段，更快采样）
- `lambda_coeff_a/b`: 控制跳步大小的Lambda调度参数，影响总步数和实际长度
- `stride`: 固定步长模式下的步长间隔

**实际长度计算公式**：
- 实际长度 = (大步阶段采样点数 × LSstepsize) + (精炼阶段采样点数 × RFstepsize)
- 其中 LSstepsize=0.45, RFstepsize=0.25
- 所有策略的实际长度控制在15-30之间，总步数控制在50-150之间

## 建议

- **首次使用**: 推荐从 `sampling_balanced.yml` 开始
- **追求质量**: 使用 `sampling_conservative.yml`
- **追求速度**: 使用 `sampling_aggressive.yml`
- **平衡选择**: 使用 `sampling_moderate.yml`

