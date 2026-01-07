#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动上传脚本 - 只上传 .py, .md, .yml 文件到 GitHub
使用方法: python3 auto_upload.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_URL = "https://github.com/milkyway21/DIffDynamic.git"
REPO_DIR = Path(__file__).parent.absolute()

def run_command(cmd, check=True, capture_output=False):
    """执行 shell 命令"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True, cwd=REPO_DIR)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check, cwd=REPO_DIR)
            return None
    except subprocess.CalledProcessError as e:
        if check:
            print(f"错误: 命令执行失败: {cmd}")
            print(f"错误信息: {e}")
            sys.exit(1)
        return None

def is_git_repo():
    """检查是否是 git 仓库"""
    return (REPO_DIR / ".git").exists()

def init_git_repo():
    """初始化 git 仓库"""
    if not is_git_repo():
        print("初始化 git 仓库...")
        run_command("git init")
        run_command(f"git remote add origin {REPO_URL}", check=False)
        print("Git 仓库初始化完成")
    else:
        # 确保远程仓库 URL 正确
        run_command(f"git remote set-url origin {REPO_URL}", check=False)
        run_command(f"git remote add origin {REPO_URL}", check=False)

def get_files_to_add():
    """获取需要添加的文件列表"""
    files_to_add = []
    
    # 需要排除的目录
    exclude_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", 
                   "data", "outputs", "pretrained_models", "docktmp", "batchsummary"}
    
    # 需要排除的文件扩展名
    exclude_extensions = {".xlsx", ".xls", ".pyc", ".pyo", ".pyd"}
    
    # 遍历所有文件
    for root, dirs, files in os.walk(REPO_DIR):
        # 排除不需要的目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        root_path = Path(root)
        # 跳过排除的目录
        if any(exclude_dir in root_path.parts for exclude_dir in exclude_dirs):
            continue
            
        for file in files:
            file_path = root_path / file
            
            # 跳过排除的扩展名
            if file_path.suffix.lower() in exclude_extensions:
                continue
            
            # 只添加 .py, .md, .yml, .yaml 文件
            if file_path.suffix.lower() in {".py", ".md", ".yml", ".yaml"}:
                # 获取相对路径
                rel_path = file_path.relative_to(REPO_DIR)
                files_to_add.append(str(rel_path))
    
    return files_to_add

def main():
    """主函数"""
    print("=" * 50)
    print("开始自动上传代码到 GitHub")
    print("=" * 50)
    print()
    
    # 初始化 git 仓库
    init_git_repo()
    
    # 获取需要添加的文件
    print("查找文件...")
    print("-" * 50)
    files_to_add = get_files_to_add()
    
    if not files_to_add:
        print("没有找到需要上传的文件")
        return
    
    print(f"找到 {len(files_to_add)} 个文件需要上传")
    
    # 添加文件
    print()
    print("添加文件到 git...")
    print("-" * 50)
    for file_path in files_to_add:
        try:
            run_command(f'git add "{file_path}"', check=False)
            print(f"  ✓ {file_path}")
        except Exception as e:
            print(f"  ✗ {file_path} - 添加失败: {e}")
    
    # 检查是否有变更
    try:
        status_output = run_command("git status --short", check=False, capture_output=True)
        if not status_output or not status_output.strip():
            print()
            print("没有需要提交的变更")
            return
    except:
        # 如果检查失败，继续尝试提交
        pass
    
    # 提交变更
    print()
    print("提交变更...")
    print("-" * 50)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = f"Auto update: {timestamp}"
    
    try:
        run_command(f'git commit -m "{commit_msg}"', check=False)
        print(f"提交信息: {commit_msg}")
    except Exception as e:
        print(f"提交失败: {e}")
        print("可能没有变更需要提交")
        return
    
    # 推送到 GitHub
    print()
    print("推送到 GitHub...")
    print("-" * 50)
    
    # 获取当前分支
    try:
        branch = run_command("git branch --show-current", capture_output=True) or "master"
    except:
        branch = "master"
        run_command("git checkout -b master", check=False)
    
    try:
        run_command(f"git push -u origin {branch}")
        print()
        print("=" * 50)
        print("✓ 代码已成功上传到 GitHub")
        print(f"  分支: {branch}")
        print(f"  时间: {timestamp}")
        print("=" * 50)
    except Exception as e:
        print()
        print("=" * 50)
        print("⚠ 推送失败，可能需要手动处理")
        print("  请检查网络连接和 GitHub 权限")
        print(f"  错误信息: {e}")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()

