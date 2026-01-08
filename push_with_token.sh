#!/bin/bash
# 使用 Personal Access Token 推送的脚本
# 使用方法: ./push_with_token.sh YOUR_TOKEN

if [ -z "$1" ]; then
    echo "使用方法: $0 <YOUR_GITHUB_TOKEN>"
    echo ""
    echo "如果没有 token，请访问: https://github.com/settings/tokens"
    echo "创建新 token，勾选 'repo' 权限"
    exit 1
fi

TOKEN=$1
REPO_URL="https://github.com/milkyway21/DiffDynamic.git"

cd "$(dirname "$0")"

# 使用 token 推送
git push https://${TOKEN}@github.com/milkyway21/DIffDynamic.git master

