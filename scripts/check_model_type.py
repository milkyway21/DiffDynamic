"""
检查检查点文件中的模型配置，确定模型类型
"""
import torch
import argparse
import sys
from pathlib import Path

# 将仓库根目录加入 sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def check_model_type(checkpoint_path):
    """检查检查点中的模型类型
    
    Args:
        checkpoint_path: 检查点文件路径
    """
    print(f"正在检查检查点: {checkpoint_path}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查模型配置
        if 'config' not in ckpt:
            print("❌ 错误：检查点中没有 'config' 键")
            return
        
        config = ckpt['config']
        if 'model' not in config:
            print("❌ 错误：配置中没有 'model' 键")
            return
        
        model_cfg = config.model
        
        # 获取模型名称
        model_name = getattr(model_cfg, 'name', 'score')
        print(f"\n📋 模型配置信息：")
        print(f"  - model.name: {model_name}")
        print(f"  - model.name (lower): {model_name.lower()}")
        
        # 判断模型类型
        model_name_lower = model_name.lower()
        if model_name_lower in ('glintdm', 'diffdynamic'):
            print(f"\n✅ 模型类型：DiffDynamic (支持 unified 模式)")
            print(f"   当前配置名称 '{model_name}' 会被识别为 DiffDynamic")
        else:
            print(f"\n⚠️  模型类型：ScorePosNet3D (不支持 unified 模式)")
            print(f"   当前配置名称 '{model_name}' 会被识别为 ScorePosNet3D")
            print(f"\n💡 解决方案：")
            print(f"   1. 如果这确实是 DiffDynamic 模型，需要修改检查点中的 model.name")
            print(f"   2. 或者修改采样脚本强制使用 DiffDynamic 类")
        
        # 检查是否有 dynamic_sample_diffusion 相关的权重
        if 'model' in ckpt:
            state_dict = ckpt['model']
            has_dynamic_methods = any(
                'dynamic_sample_diffusion' in key or 
                'dynamic_large_step_defaults' in key or
                'dynamic_refine_defaults' in key
                for key in state_dict.keys()
            )
            
            if has_dynamic_methods:
                print(f"\n🔍 检查点权重中包含动态采样相关参数")
            else:
                print(f"\n🔍 检查点权重中未发现动态采样相关参数（这是正常的，这些是方法而非权重）")
        
        # 显示更多配置信息
        print(f"\n📝 其他模型配置：")
        print(f"  - use_grad_fusion: {getattr(model_cfg, 'use_grad_fusion', 'N/A')}")
        print(f"  - ligand_v_input: {getattr(model_cfg, 'ligand_v_input', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 错误：无法加载检查点文件")
        print(f"   错误信息：{e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='检查检查点文件中的模型类型')
    parser.add_argument('checkpoint', type=str, help='检查点文件路径')
    args = parser.parse_args()
    
    check_model_type(args.checkpoint)
























