#!/bin/bash

# ============= 配置部分（请根据需求修改） =============

# 1. nohup 会话名称（可选，用于标识任务）
SESSION_NAME="process_criteo"

# 2. 日志文件保存路径（可自定义）
LOG_PATH="./logs/${SESSION_NAME}_$(date +\%Y\%m\%d_\%H\%M\%S).log"

# 3. 是否启用日志轮转（可选，默认关闭）
ENABLE_LOG_ROTATE=false

# ============= 配置部分结束 ========================

# 创建日志目录（如果不存在）
mkdir -p "$(dirname "$LOG_PATH")"

# 记录执行命令到日志文件
echo "========================================" >> "$LOG_PATH"
echo "执行时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_PATH"
echo "会话名称: $SESSION_NAME" >> "$LOG_PATH"
echo "执行命令: Python data processing pipeline" >> "$LOG_PATH"
echo "========================================" >> "$LOG_PATH"
echo "" >> "$LOG_PATH"

# 启动命令（后台运行 + 日志重定向）
# 使用 heredoc 方式传递 Python 代码，避免引号嵌套问题
# 添加 -u 参数强制 Python 使用无缓冲模式，确保实时输出到日志
nohup python -u <<'EOF' >> "$LOG_PATH" 2>&1 &

from dataprocess.criteo_optimized import preprocess_criteo_dataset
feature_size = preprocess_criteo_dataset(
    source_path='/data/oss_bucket_0/ssrnet/pub_data/criteo/criteo.data',
    output_path='./data/Criteo/',
    verbose=1
)
print(f'Criteo processing completed with {feature_size:,} total features')

from dataprocess.kfold_split import create_stratified_splits
from dataprocess.config import CriteoConfig
config = CriteoConfig(data_path='./data/Criteo/')
create_stratified_splits(config)

from dataprocess.base import DataScaler
from dataprocess.config import CriteoConfig
config = CriteoConfig(data_path='./data/Criteo/')
numerical_columns = list(range(13))  # Criteo前13列为数值特征
DataScaler.scale_data_parts(config, numerical_columns, scale_method='log')


# from dataprocess.aliccp_optimized import preprocess_aliccp_dataset
# feature_size = preprocess_aliccp_dataset(
#     source_path='/data/oss_bucket_0/ssrnet/pub_data/ali-ccp/',
#     output_path='./data/Aliccp/',
#     process_test=False,
#     use_log_scaling=True,
#     verbose=1
# )
# print(f'Ali-CCP processing completed with {feature_size:,} total features')

# from dataprocess.kfold_split import create_stratified_splits
# from dataprocess.config import AliccpConfig
# config = AliccpConfig(data_path='./data/Aliccp/')
# create_stratified_splits(config)
# print(f'Ali-CCP kfold_split completed !')

EOF

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
