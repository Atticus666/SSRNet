#!/bin/bash

# ============= 配置部分（请根据需求修改） =============

# 1. nohup 会话名称（可选，用于标识任务）
SESSION_NAME="avazu_ssrnet_t"
# 2. 训练命令
CMD="
python runners/train_ssrnet_t.py \
    --data avazu \
    --data_path ./data/ \
    --block_version t21 \
    --embedding_size 16 \
    --tokennum_list 8 8 \
    --hidden_unit_list 128 128 \
    --top_k_list 128 128 \
    --out_unit_list 128 128 \
    --iterations 3 \
    --alpha_init 0.3 0.3 \
    --scale_init 1.0 1.0 \
    --use_ssr_linear False \
    --use_gate True \
    --use_block_dense True \
    --use_block_mean_pooling True \
    --dropout_rates 0.0 0.0 0.0 \
    --l2_reg 0.0 \
    --batch_size 1024 \
    --epoch 3 \
    --optimizer_type adam \
    --learning_rate 0.001 \
    --num_runs 1 \
    --save_path ./checkpoint/${SESSION_NAME}/ \
    --is_save 0 \
    --verbose 1
"


# 3. 日志文件保存路径（可自定义）
LOG_PATH="./logs/$(date +\%Y\%m\%d_\%H\%M\%S)_${SESSION_NAME}.log"



# 4. 是否启用日志轮转（可选，默认关闭）
ENABLE_LOG_ROTATE=false
# ============= 配置部分结束 ========================


# 创建日志目录（如果不存在）
mkdir -p "$(dirname "$LOG_PATH")"

# 记录执行命令到日志文件
echo "========================================" >> "$LOG_PATH"
echo "执行时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_PATH"
echo "会话名称: $SESSION_NAME" >> "$LOG_PATH"
echo "执行命令:" >> "$LOG_PATH"
printf "%s\n" "$CMD" >> "$LOG_PATH"
echo "========================================" >> "$LOG_PATH"
echo "" >> "$LOG_PATH"

# 启动命令（后台运行 + 日志重定向）
nohup $CMD >> "$LOG_PATH" 2>&1 &

# 解除与当前终端的关联（确保关闭终端后进程仍在运行）
disown

# 可选：日志轮转（每天0点切割日志）
if [ "$ENABLE_LOG_ROTATE" = true ]; then
    echo "Setting up log rotation..."
    LOG_DIR=$(dirname "$LOG_PATH")
    LOG_BASE=$(basename "$LOG_PATH" .log)
    (crontab -l 2>/dev/null; echo "0 0 * * * find $LOG_DIR -name \"${LOG_BASE}_*.log\" -mtime +7 -exec rm {} \;" ) | crontab -
fi

echo "任务 [${SESSION_NAME}] 已启动，日志保存至: $LOG_PATH"
