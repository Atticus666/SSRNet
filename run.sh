#!/bin/bash

# ============= Configuration Section (Modify as needed) =============

# 1. nohup session name (optional, used to identify the task)
SESSION_NAME="criteo_ssrnet_t_i3_a1_s1"
# 2. Training command
CMD="
python runners/train_ssrnet_t.py \
    --data criteo \
    --data_path /data/oss_bucket_0/ssrnet/data/ \
    --block_version t21 \
    --embedding_size 16 \
    --tokennum_list 8 8 \
    --hidden_unit_list 128 128 \
    --top_k_list 128 128 \
    --out_unit_list 128 128 \
    --iterations 3 \
    --alpha_init 1.0 1.0 \
    --scale_init 1.0 1.0 \
    --use_ssr_linear False \
    --use_block_dense True \
    --use_block_mean_pooling False \
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


# 3. Log file save path (customizable)
LOG_PATH="./logs/$(date +\%Y\%m\%d_\%H\%M\%S)_${SESSION_NAME}.log"



# 4. Enable log rotation (optional, disabled by default)
ENABLE_LOG_ROTATE=false
# ============= End of Configuration Section ========================


# Create log directory (if it doesn't exist)
mkdir -p "$(dirname "$LOG_PATH")"

# Record execution command to log file
echo "========================================" >> "$LOG_PATH"
echo "Execution time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_PATH"
echo "Session name: $SESSION_NAME" >> "$LOG_PATH"
echo "Execution command:" >> "$LOG_PATH"
printf "%s\n" "$CMD" >> "$LOG_PATH"
echo "========================================" >> "$LOG_PATH"
echo "" >> "$LOG_PATH"

# Start command (background execution + log redirection)
nohup $CMD >> "$LOG_PATH" 2>&1 &

# Disown from current terminal (ensure process continues after terminal closes)
disown

# Optional: Log rotation (rotate logs at midnight daily)
if [ "$ENABLE_LOG_ROTATE" = true ]; then
    echo "Setting up log rotation..."
    LOG_DIR=$(dirname "$LOG_PATH")
    LOG_BASE=$(basename "$LOG_PATH" .log)
    (crontab -l 2>/dev/null; echo "0 0 * * * find $LOG_DIR -name \"${LOG_BASE}_*.log\" -mtime +7 -exec rm {} \;" ) | crontab -
fi

echo "Task [${SESSION_NAME}] has been started, log saved to: $LOG_PATH"
