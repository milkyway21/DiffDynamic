# 修复 OpenBabel GLIBCXX 版本问题

## 问题描述

错误信息：
```
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found 
(required by /opt/conda/envs/diffdynamic/lib/python3.8/site-packages/openbabel/_openbabel.so)
```

**原因**：系统的 `libstdc++.so.6` 版本太旧，不包含 OpenBabel 需要的 `GLIBCXX_3.4.29`。需要优先使用 conda 环境中的更新版本。

## 永久修复方案（推荐）

### 方法 1：使用自动修复脚本（最简单）

在 Docker 容器中运行：

```bash
# 1. 进入工作目录
cd /workspace

# 2. 运行修复脚本
bash fix_openbabel_glibcxx.sh

# 3. 重新激活 conda 环境
conda deactivate
conda activate diffdynamic

# 4. 验证修复
python3 -c "import openbabel; print('✅ OpenBabel 导入成功')"
```

### 方法 2：手动创建激活脚本

如果自动脚本不工作，可以手动创建：

```bash
# 1. 创建激活脚本目录
mkdir -p /opt/conda/envs/diffdynamic/etc/conda/activate.d

# 2. 创建激活脚本
cat > /opt/conda/envs/diffdynamic/etc/conda/activate.d/fix_glibcxx.sh << 'EOF'
#!/bin/bash
# 确保 conda 环境的 lib 目录优先
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi
EOF

# 3. 设置执行权限
chmod +x /opt/conda/envs/diffdynamic/etc/conda/activate.d/fix_glibcxx.sh

# 4. 重新激活环境
conda deactivate
conda activate diffdynamic
```

### 方法 3：在 Dockerfile 中永久修复

如果需要在 Docker 镜像中永久修复，可以在 Dockerfile 中添加：

```dockerfile
# 修复 OpenBabel GLIBCXX 问题
RUN mkdir -p /opt/conda/envs/diffdynamic/etc/conda/activate.d && \
    echo '#!/bin/bash' > /opt/conda/envs/diffdynamic/etc/conda/activate.d/fix_glibcxx.sh && \
    echo 'if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then' >> /opt/conda/envs/diffdynamic/etc/conda/activate.d/fix_glibcxx.sh && \
    echo '    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' >> /opt/conda/envs/diffdynamic/etc/conda/activate.d/fix_glibcxx.sh && \
    echo 'fi' >> /opt/conda/envs/diffdynamic/etc/conda/activate.d/fix_glibcxx.sh && \
    chmod +x /opt/conda/envs/diffdynamic/etc/conda/activate.d/fix_glibcxx.sh
```

## 验证修复

运行以下命令验证修复是否成功：

```bash
# 1. 检查 LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH | grep -o '/opt/conda/envs/diffdynamic/lib' && echo "✅ Conda lib 在路径中"

# 2. 测试 OpenBabel 导入
python3 -c "from openbabel import openbabel as ob; print('✅ OpenBabel 导入成功')"

# 3. 检查使用的 libstdc++
python3 -c "import openbabel; import os; so_file = os.path.join(os.path.dirname(openbabel.__file__), 'openbabel', '_openbabel.so'); import subprocess; result = subprocess.run(['ldd', so_file], capture_output=True, text=True); print(result.stdout)" | grep libstdc
```

## 其他解决方案

### 方案 A：更新系统 libstdc++（不推荐）

如果系统允许，可以更新系统的 libstdc++，但这可能影响其他程序。

### 方案 B：重新编译 OpenBabel（复杂）

使用与系统兼容的 libstdc++ 重新编译 OpenBabel，但过程复杂。

### 方案 C：使用 conda 的 libstdcxx-ng 包

确保安装了最新版本：

```bash
conda install -c conda-forge libstdcxx-ng -y
```

## 注意事项

1. **每次激活环境时自动应用**：使用激活脚本后，每次运行 `conda activate diffdynamic` 时都会自动设置正确的 `LD_LIBRARY_PATH`。

2. **Docker 容器重启**：如果 Docker 容器重启，激活脚本仍然有效，因为它是 conda 环境的一部分。

3. **多用户环境**：如果多个用户使用同一个 conda 环境，修复对所有用户都有效。

4. **优先级**：确保 conda 环境的 lib 目录在系统目录**之前**，这样会优先使用 conda 的 libstdc++。

## 故障排除

如果修复后仍然有问题：

1. **检查脚本是否执行**：
   ```bash
   ls -la /opt/conda/envs/diffdynamic/etc/conda/activate.d/
   ```

2. **手动设置环境变量**：
   ```bash
   export LD_LIBRARY_PATH=/opt/conda/envs/diffdynamic/lib:$LD_LIBRARY_PATH
   python3 -c "import openbabel; print('测试')"
   ```

3. **检查 libstdc++ 版本**：
   ```bash
   strings /opt/conda/envs/diffdynamic/lib/libstdc++.so.6 | grep GLIBCXX | tail -5
   ```

4. **查看详细错误**：
   ```bash
   python3 -c "import openbabel" 2>&1 | head -20
   ```
