#!/bin/bash

# ================= 🔐 您的 Kaggle 凭证 (直接填在这里) =================
# 您的用户名 (已帮您填好)
MY_USERNAME="canhan"

# 🔴 请将您的 Key 粘贴在下面引号内 (把原来的文字删掉)
MY_KEY="KGAT_b5f9845c1e20b28db2332edc0bd77c8f"

# ================= ⚙️ 下载配置 =================
DATASET_ID="samirrana00/ninapro-db6"
TARGET_DIR="/bigdata/emgdata_public/DB_raw/NinaPro_DB6"
ZIP_NAME="ninapro_db6_full.zip"
# ===================================================================

# 修复点：将 == 改为 =，兼容 zsh
if [ "$MY_KEY" = "在这里粘贴您的Key" ]; then
    echo "❌ 错误: 您忘记在脚本里填入 Kaggle Key 了！"
    echo "请使用 nano 编辑此脚本，修改 MY_KEY 这一行。"
    exit 1
fi

# 检查 aria2c
if ! command -v aria2c &> /dev/null; then
    echo "❌ 错误: 未安装 aria2c (apt install aria2)。"
    exit 1
fi

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit

echo "🚀 开始任务: Kaggle 直链极速下载"
echo "👤 用户: $MY_USERNAME"

# --- 第一步: 生成 Python 脚本 (直接注入您的 Key) ---
cat <<EOF > get_direct_url.py
import requests
import sys

# 直接使用脚本头部定义的变量
username = "$MY_USERNAME"
key = "$MY_KEY"
dataset = "$DATASET_ID"

api_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"

try:
    # 禁止重定向，获取 Location 头
    r = requests.get(api_url, auth=(username, key), allow_redirects=False)
    
    if r.status_code == 302:
        print(r.headers['Location'])
    elif r.status_code == 403:
        print("Error:403_Forbidden (Key错误或数据集不可访问)")
        sys.exit(1)
    elif r.status_code == 401:
        print("Error:401_Unauthorized (账号或Key错误)")
        sys.exit(1)
    else:
        print(f"Error:StatusCode_{r.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"Error:{e}")
    sys.exit(1)
EOF

# --- 第二步: 获取直链 ---
echo "🕵️ 正在向 Kaggle 申请直链..."
DIRECT_URL=$(python3 get_direct_url.py)

# 错误处理
if [[ "$DIRECT_URL" == Error* ]] || [[ -z "$DIRECT_URL" ]]; then
    echo "❌ 获取失败: $DIRECT_URL"
    echo "请检查脚本顶部的 MY_KEY 是否粘贴正确。"
    rm get_direct_url.py
    exit 1
fi

echo "✅ 成功获取！启动 Aria2c 16线程下载..."
rm get_direct_url.py

# --- 第三步: 极速下载 ---
aria2c -x 16 -s 16 -k 1M -c \
       -o "$ZIP_NAME" \
       "$DIRECT_URL"

# --- 第四步: 解压 ---
if [ -f "$ZIP_NAME" ]; then
    echo "📦 正在解压..."
    unzip -q "$ZIP_NAME"
    if [ $? -eq 0 ]; then
        echo "🎉 全部完成！数据都在这里了。"
    else
        echo "❌ 解压失败。"
    fi
else
    echo "❌ 下载未完成。"
fi