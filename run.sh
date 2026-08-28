#!/bin/bash
set -e

echo "===== ITU Track1 Space Inference ====="
echo "当前工作目录: $(pwd)"
echo "Python 版本: $(python --version)"

# 输出目录：验证/上星环境用 /output，开发环境同步到 /home/spaceapp/project/data
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "$OUTPUT_DIR"

# 数据目录：优先使用解压后的 /home/spaceapp/project/source，否则回退 /rs
DATA_DIR="${DATA_DIR:-/home/spaceapp/project/source}"
mkdir -p "$DATA_DIR"

# 如果 /rs 下有 track1_data.zip，先复制到可写目录再解压
echo "===== 检查并准备输入数据 ====="
if [ -f "/rs/track1_data.zip" ]; then
    echo "发现 /rs/track1_data.zip，复制到 $DATA_DIR ..."
    cp /rs/track1_data.zip "$DATA_DIR/track1_data.zip"
    echo "解压中..."
    cd "$DATA_DIR"
    unzip -o track1_data.zip
    cd -
    echo "解压完成，影像文件数量: $(ls "$DATA_DIR"/*.tif "$DATA_DIR"/*.tiff 2>/dev/null | wc -l)"
elif [ -d "/rs" ]; then
    echo "/rs 已挂载，使用 /rs 作为数据目录"
    DATA_DIR="/rs"
else
    echo "警告: /rs 未挂载且未找到 track1_data.zip"
fi

echo "数据目录: $DATA_DIR"
ls -la "$DATA_DIR" | head -n 20

# 检查模型
echo "===== 检查模型文件 ====="
ls -la /workspace/models/

# 执行推理
echo "===== 开始推理 ====="
DATA_DIR="$DATA_DIR" OUTPUT_DIR="$OUTPUT_DIR" \
    python /workspace/code-server/inference_cascade_lstm_v90-4-5-tau14-debug.py

# 开发环境下同时复制结果到平台可下载目录
if [ -d "/home/spaceapp/project/data" ] && [ "$OUTPUT_DIR" != "/home/spaceapp/project/data" ]; then
    echo "===== 复制结果到开发环境下载目录 ====="
    cp -v "$OUTPUT_DIR"/result.json /home/spaceapp/project/data/result.json 2>/dev/null || true
fi

# 检查结果
echo "===== 推理结束，输出文件 ====="
ls -la "$OUTPUT_DIR"

echo "===== 完成 ====="
