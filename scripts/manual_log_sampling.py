"""
手动导入采样记录到Excel日志的脚本。

用法:
    python scripts/manual_log_sampling.py
"""

import os
import ast
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.sampling_recorder import log_sampling_record

# 从日志中提取的采样参数
raw = """{'model': {'checkpoint': './pretrained_models/pretrained_GlintDM.pt', 'use_grad_fusion': True, 'grad_fusion_lambda': None, 'mode': 'linear', 'start': 0.8, 'end': 0.2}, 'sample': {'seed': 2021, 'num_samples': 100, 'num_steps': 1000, 'pos_only': False, 'center_pos_mode': 'protein', 'sample_num_atoms': 'prior', 'mode': 'dynamic', 'dynamic': {'method': 'auto', 'large_step': {'schedule': 'lambda', 'batch_size': 4, 'stride': 40, 'step_size': 1.0, 'n_repeat': 4, 'noise_scale': 0.0, 'time_lower': 500}, 'refine': {'schedule': 'lambda', 'stride': 10, 'step_size': 0.2, 'time_upper': 500, 'time_lower': 0, 'cycles': 1, 'n_sampling': 2, 'noise_scale': 0.0}, 'selector': {'top_n': 12, 'min_qed': 0.0, 'max_sa': 6.0, 'qed_weight': 1.0, 'sa_weight': 1.0}}}}"""

if __name__ == '__main__':
    # 解析参数字典
    params = ast.literal_eval(raw)
    
    # 创建输出目录
    result_dir = 'outputs/manual_import'
    os.makedirs(result_dir, exist_ok=True)
    
    # 记录采样信息
    success = log_sampling_record(
        params=params,
        result_dir=result_dir,
        sampling_mode=params['sample'].get('mode', 'unknown'),
        result_file='outputs/manual_import/result_manual.pt',
        extra_info={'source': 'manual_log'}
    )
    
    if success:
        print(f'✓ 采样记录已成功添加到 outputs/sampling_history.xlsx')
    else:
        print(f'✗ 采样记录添加失败')
        sys.exit(1)



























