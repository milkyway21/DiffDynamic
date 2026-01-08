#!/bin/bash
# 设置 SSH 密钥用于 GitHub 推送
# 使用方法: ./setup_ssh.sh

echo "=========================================="
echo "GitHub SSH 密钥设置"
echo "=========================================="
echo ""

# 检查是否已有 SSH 密钥
if [ -f ~/.ssh/id_rsa.pub ] || [ -f ~/.ssh/id_ed25519.pub ]; then
    echo "检测到已存在的 SSH 密钥"
    if [ -f ~/.ssh/id_ed25519.pub ]; then
        KEY_FILE=~/.ssh/id_ed25519.pub
    else
        KEY_FILE=~/.ssh/id_rsa.pub
    fi
    echo "公钥文件: $KEY_FILE"
    echo ""
    echo "请将以下公钥添加到 GitHub:"
    echo "1. 访问: https://github.com/settings/keys"
    echo "2. 点击 'New SSH key'"
    echo "3. 复制下面的公钥内容并粘贴"
    echo ""
    echo "----------------------------------------"
    cat "$KEY_FILE"
    echo "----------------------------------------"
else
    echo "生成新的 SSH 密钥..."
    echo ""
    read -p "请输入你的 GitHub 邮箱 (直接回车使用默认): " email
    if [ -z "$email" ]; then
        email="milkyway21@users.noreply.github.com"
    fi
    
    # 生成 SSH 密钥
    ssh-keygen -t ed25519 -C "$email" -f ~/.ssh/id_ed25519 -N ""
    
    echo ""
    echo "SSH 密钥已生成！"
    echo ""
    echo "请将以下公钥添加到 GitHub:"
    echo "1. 访问: https://github.com/settings/keys"
    echo "2. 点击 'New SSH key'"
    echo "3. 标题填写: server01 (或任意名称)"
    echo "4. 复制下面的公钥内容并粘贴"
    echo ""
    echo "----------------------------------------"
    cat ~/.ssh/id_ed25519.pub
    echo "----------------------------------------"
fi

echo ""
read -p "按回车键继续测试 SSH 连接..."
echo ""

# 测试 SSH 连接
echo "测试 SSH 连接..."
ssh -T git@github.com 2>&1 | head -5

echo ""
echo "如果看到 'Hi milkyway21! You've successfully authenticated'，说明配置成功！"
echo ""
echo "现在可以使用以下命令推送代码:"
echo "  git push -u origin master"

