# 总结：
# - 从给定 PDB 蛋白口袋生成空配体图，调用扩散模型采样配体结构并重建分子。
# - 使用配置中给定的训练权重与采样参数，可选指定样本数与输出目录。
# - 对重建成功的分子保存为 SDF，同时持久化采样结果和配置，便于评估与复现。

import argparse  # 导入 argparse，解析命令行参数。
import os  # 导入 os，用于路径操作。
import shutil  # 导入 shutil，复制配置文件。
import sys  # 修改 sys.path，确保可以引用项目内模块。
import subprocess  # 导入 subprocess，用于执行外部脚本。
from datetime import datetime  # 导入 datetime，用于生成时间戳文件名。
from pathlib import Path  # 方便地解析仓库根目录。

# WSL CUDA 支持：设置库路径（如果存在）
wsl_cuda_lib = Path("/usr/lib/wsl/lib")
if wsl_cuda_lib.exists() and str(wsl_cuda_lib) not in os.environ.get("LD_LIBRARY_PATH", ""):
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{wsl_cuda_lib}:{current_ld_path}" if current_ld_path else str(wsl_cuda_lib)

# 将仓库根目录加入 sys.path，防止相对运行脚本时找不到 utils 等模块。
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 设置 PyTorch CUDA 库路径（必须在导入 torch 之前）
# 这确保 torch-cluster、torch-scatter 等扩展可以找到 libc10_cuda.so
# 通过查找 site-packages 中的 torch 包来定位 lib 目录
import site
torch_lib_paths = []
for site_pkg in site.getsitepackages():
    torch_lib = Path(site_pkg) / "torch" / "lib"
    if torch_lib.exists():
        torch_lib_paths.append(str(torch_lib))
        break

# 如果找到 PyTorch lib 目录，添加到 LD_LIBRARY_PATH
if torch_lib_paths:
    torch_lib_str = torch_lib_paths[0]
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if torch_lib_str not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_str}:{current_ld_path}" if current_ld_path else torch_lib_str

import torch  # 导入 PyTorch，用于张量处理和模型推断。
from torch_geometric.transforms import Compose  # 导入转换组合工具。

import utils.misc as misc  # 导入通用工具（日志、配置等）。
import utils.transforms as trans  # 导入特征转换模块。
from datasets.pl_data import ProteinLigandData, torchify_dict  # 导入数据结构及张量化工具。
from models.molopt_score_model import ScorePosNet3D, DiffDynamic  # 导入扩散模型类。
from scripts.sample_diffusion import sample_diffusion_ligand, sample_dynamic_diffusion_ligand  # 导入采样函数。
from utils.data import PDBProtein  # 导入 PDB 解析工具。
from utils import reconstruct  # 导入分子重建工具。
from rdkit import Chem  # 导入 RDKit，写出分子文件。


def pdb_to_pocket_data(pdb_path):  # 将 PDB 文件转换为无配体的口袋图。
    """读取蛋白 PDB 文件并构造仅含蛋白节点的 `ProteinLigandData` 对象。"""
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()  # 解析蛋白坐标与原子属性。
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),  # 张量化蛋白数据。
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 构建参数解析器。
    parser.add_argument('config', type=str)  # 配置文件路径。
    parser.add_argument('--pdb_path', type=str, required=True)  # PDB 文件路径。
    parser.add_argument('--device', type=str, default='cuda:0')  # 指定运行设备。
    parser.add_argument('--num_samples', type=int, default=100)  # 生成样本数量。
    parser.add_argument('--batch_size', type=int, default=100)  # 批量大小（用于 baseline 模式）。
    parser.add_argument('--result_path', type=str, default='./outputs')  # 结果输出目录。
    parser.add_argument('--mode', type=str, choices=['baseline', 'dynamic'], default=None)  # 采样模式。
    args = parser.parse_args()  # 解析命令行参数。

    logger = misc.get_logger('sampling')  # 创建日志器。
    logger.info('=' * 50)
    logger.info('Starting sampling from PDB file...')
    logger.info(f'PDB path: {args.pdb_path}')
    logger.info(f'Device: {args.device}')
    logger.info(f'Number of samples: {args.num_samples}')
    logger.info('=' * 50)

    # 加载配置文件：从 YAML 文件读取采样配置。
    config = misc.load_config(args.config)  # 加载采样配置。
    logger.info(f'Config loaded: {config}')  # 记录配置信息到日志。
    misc.seed_all(config.sample.seed)  # 设置随机种子，确保可复现性。

    # 更新 num_samples（如果命令行指定了）
    if args.num_samples:
        config.sample.num_samples = args.num_samples

    # 加载模型检查点：从检查点文件恢复模型权重和训练配置。
    logger.info(f'Loading model checkpoint from: {config.model.checkpoint}')
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)  # 加载检查点到指定设备。
    logger.info(f"Training Config loaded successfully")  # 记录训练配置信息。

    # 初始化特征转换器：创建蛋白和配体的特征提取管道。
    protein_featurizer = trans.FeaturizeProteinAtom()  # 创建蛋白原子特征化器。
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode  # 从检查点读取配体原子编码模式。
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)  # 创建配体原子特征化器。
    transform = Compose([  # 组合多个转换器为单一管道。
        protein_featurizer,  # 蛋白特征转换。
        ligand_featurizer,  # 配体特征转换。
        trans.FeaturizeLigandBond(),  # 配体键特征转换。
    ])

    # 从 PDB 文件加载数据
    logger.info(f'Loading PDB file: {args.pdb_path}')
    data = pdb_to_pocket_data(args.pdb_path)  # 从 PDB 文件创建数据对象。
    data = transform(data)  # 应用特征转换。
    logger.info(f'PDB file loaded successfully. Protein atoms: {data.protein_pos.shape[0]}')

    # 加载模型：根据检查点配置实例化并加载模型权重。
    model_cfg = ckpt['config'].model  # 从检查点读取模型配置。
    model_name = getattr(model_cfg, 'name', 'score').lower()  # 获取模型名称，默认为 'score'。
    
    # 支持从配置强制指定模型类型（用于处理检查点配置名称不匹配的情况）
    force_model_type = None
    if hasattr(config, 'model') and hasattr(config.model, 'force_model_type'):
        force_model_type = config.model.force_model_type.lower()
    
    if force_model_type:
        if force_model_type == 'diffdynamic':
            model_cls = DiffDynamic
            logger.info(f'强制使用 DiffDynamic 模型（根据配置 force_model_type）')
        elif force_model_type == 'scoreposnet3d':
            model_cls = ScorePosNet3D
            logger.info(f'强制使用 ScorePosNet3D 模型（根据配置 force_model_type）')
        else:
            logger.warning(f'未知的 force_model_type: {force_model_type}，使用默认判断')
            model_cls = DiffDynamic if model_name in ('glintdm', 'diffdynamic') else ScorePosNet3D
    else:
        # 支持 glintdm 和 diffdynamic 两种配置值（向后兼容）
        model_cls = DiffDynamic if model_name in ('glintdm', 'diffdynamic') else ScorePosNet3D  # 根据名称选择模型类。
    
    logger.info(f'检查点中的 model.name: {model_name}')
    logger.info(f'最终使用的 Model class: {model_cls.__name__}')
    model = model_cls(  # 实例化模型。
        model_cfg,  # 模型配置。
        protein_atom_feature_dim=protein_featurizer.feature_dim,  # 蛋白原子特征维度。
        ligand_atom_feature_dim=ligand_featurizer.feature_dim  # 配体原子特征维度。
    ).to(args.device)  # 将模型移动到指定设备。
    logger.info('Loading model weights...')
    model.load_state_dict(ckpt['model'])  # 加载模型权重。
    logger.info(f'Model loaded successfully! Checkpoint: {config.model.checkpoint}')

    # 确定采样模式：优先使用命令行参数，否则使用配置文件
    sampling_mode = args.mode or config.sample.get('mode', 'baseline')  # 决定采样模式。
    logger.info(f'Sampling mode: {sampling_mode}')
    if sampling_mode not in ['baseline', 'dynamic']:
        raise ValueError(f'Unsupported sampling mode: {sampling_mode}')

    if sampling_mode == 'dynamic':  # 动态采样模式。
        logger.info('Using dynamic sampling mode (unified/legacy based on config)')
        # 确保动态采样配置存在，并设置默认值。
        config.sample.setdefault('dynamic', {})  # 如果不存在则创建空字典。
        config.sample.dynamic.setdefault('large_step', {})  # 确保大步配置存在。
        config.sample.dynamic['large_step'].setdefault('batch_size', args.batch_size)  # 设置大步批量大小。
        # 打印采样配置中的动态采样参数，用于调试
        if 'dynamic' in config.sample:
            logger.info(f'Sampling config - large_step: {config.sample.dynamic.get("large_step", {})}')
            logger.info(f'Sampling config - refine: {config.sample.dynamic.get("refine", {})}')
        # 执行动态采样。
        logger.info('Starting dynamic sampling...')
        dynamic_output = sample_dynamic_diffusion_ligand(
            model=model,
            data=data,
            config=config,
            ligand_atom_mode=ligand_atom_mode,
            device=args.device,
            logger=logger
        )
        result = {
            'data': data,
            'pred_ligand_pos': dynamic_output['pos_list'],
            'pred_ligand_v': dynamic_output['v_list'],
            'pred_ligand_pos_traj': dynamic_output['pos_traj'],
            'pred_ligand_v_traj': dynamic_output['v_traj'],
            'pred_ligand_log_v_traj': dynamic_output['log_v_traj'],
            'time': dynamic_output['time_list'],
            'meta': dynamic_output['meta'],
            'mode': 'dynamic'  # 标记采样模式。
        }
    else:  # 基线采样模式。
        logger.info('Using baseline sampling mode')
        # 执行标准扩散采样。
        logger.info('Starting baseline sampling...')
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'pred_ligand_v0_traj': pred_v0_traj,
            'pred_ligand_vt_traj': pred_vt_traj,
            'time': time_list,  # 采样耗时列表。
            'mode': 'baseline'  # 标记采样模式。
        }
    logger.info('Sampling done!')  # 记录采样完成信息。

    # 保存结果：将采样结果和配置保存到输出目录。
    result_path = args.result_path  # 获取结果输出目录路径。
    os.makedirs(result_path, exist_ok=True)  # 创建输出目录（如果不存在）。
    logger.info(f'Saving results to: {result_path}')
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))  # 备份采样配置文件。
    
    # 保存采样结果，使用日期+时间命名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(result_path, f'result_{timestamp}.pt')
    torch.save(result, result_file)  # 保存采样结果为 PyTorch 文件。
    logger.info(f'Results saved to: {result_file}')
    
    # 自动执行转换器脚本生成SDF文件
    # 使用项目根目录动态获取转换器脚本路径，支持跨平台
    converter_script = REPO_ROOT / 'targetdiff_pt_to_sdf_converter.py'
    if converter_script.exists():
        logger.info(f'Executing converter script: {converter_script}')
        # 生成SDF文件的输出目录（基于.pt文件名）
        sdf_output_dir = os.path.join(result_path, f'sdf_{timestamp}')
        try:
            # 执行转换器脚本
            subprocess.run([
                sys.executable, str(converter_script), result_file,
                '--output_dir', sdf_output_dir
            ], check=True)
            logger.info(f'SDF files generated in: {sdf_output_dir}')
        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to execute converter script: {e}')
        except Exception as e:
            logger.error(f'Error executing converter script: {e}')
    else:
        logger.warning(f'Converter script not found: {converter_script}')

    # 重建分子并保存为 SDF 文件
    logger.info('Reconstructing molecules and saving SDF files...')
    
    n_success = 0
    for idx, (pos, v) in enumerate(zip(result['pred_ligand_pos'], result['pred_ligand_v'])):
        try:
            # 重建分子
            pos_array = pos.detach().cpu().numpy() if torch.is_tensor(pos) else pos
            v_tensor = v.detach().cpu() if torch.is_tensor(v) else torch.tensor(v)
            atom_numbers = v_tensor.argmax(dim=-1).numpy() if v_tensor.dim() > 1 else v_tensor.numpy()
            
            aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)
            mol = reconstruct.reconstruct_from_generated(pos_array, atom_numbers, aromatic_flags)
            
            if mol is not None:
                sdf_path = os.path.join(result_path, f'sample_{idx:04d}.sdf')
                writer = Chem.SDWriter(sdf_path)
                writer.write(mol)
                writer.close()
                n_success += 1
        except Exception as e:
            logger.warning(f'Failed to reconstruct sample {idx}: {e}')
            continue
    
    logger.info(f'Successfully saved {n_success}/{len(result["pred_ligand_pos"])} molecules as SDF files.')
    logger.info('=' * 50)
    logger.info('All done!')
    logger.info('=' * 50)

