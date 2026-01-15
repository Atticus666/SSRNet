#!/bin/bash

# ============= Configuration Section (Modify as needed) =============

# 1. nohup session name (optional, used to identify the task)
SESSION_NAME="process_criteo"

# 2. Log file save path (customizable)
LOG_PATH="./logs/${SESSION_NAME}_$(date +\%Y\%m\%d_\%H\%M\%S).log"

# 3. Enable log rotation (optional, disabled by default)
ENABLE_LOG_ROTATE=false

# ============= End of Configuration Section ========================

# Create log directory (if it doesn't exist)
mkdir -p "$(dirname "$LOG_PATH")"

# Record execution command to log file
echo "========================================" >> "$LOG_PATH"
echo "Execution time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_PATH"
echo "Session name: $SESSION_NAME" >> "$LOG_PATH"
echo "执行命令: Python data processing pipeline" >> "$LOG_PATH"
echo "========================================" >> "$LOG_PATH"
echo "" >> "$LOG_PATH"

# Start command (background execution + log redirection)
# Use heredoc to pass Python code, avoiding quote nesting issues
# Add -u parameter to force Python unbuffered mode for real-time log output
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
numerical_columns = list(range(13))  # Criteo first 13 columns are numerical features
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
