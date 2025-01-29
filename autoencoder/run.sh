#!/bin/bash

### DO NOT CTRL+C THIS FILE
### until proper handling is done for the "nohup dstat" commands

# Variables
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('hyperparameters.json')).get('output_dir', 'output_dir'))")
UTILLOG_TRAIN_FILE="$OUTPUT_DIR/utillog_train.csv"
UTILLOG_INF_FILE="$OUTPUT_DIR/utillog_inference.csv"

# Create output directory
mkdir $OUTPUT_DIR

# Initialize training utilization logging
nohup dstat -m -c  | while read line; do echo "$(date +'%Y-%m-%d %H:%M:%S') $line" >> $UTILLOG_TRAIN_FILE; done &
DSTAT_PID=$!

# Train model
echo -e "\nTraining Model\n"
python3 train.py

if [ $? -ne 0 ]; then
    echo "Training failed, stopping program"
    exit 1
fi

# Kill logging processes
kill $DSTAT_PID

# Initialize inference utilization logging
nohup dstat -m -c  | while read line; do echo "$(date +'%Y-%m-%d %H:%M:%S') $line" >> $UTILLOG_INF_FILE; done &
DSTAT_PID=$!

# Inference model
echo -e "\nInferencing Model\n"
python3 inference.py  --model_path="./output_dir/model.keras"

# Kill logging processes
kill $DSTAT_PID

# Fix utillog training
tail -n +2 $UTILLOG_TRAIN_FILE > tempfile
sed -i 's/|/ /g' tempfile
sed -i 's/  */ /g' tempfile
awk 'NR==1{$1="date"; $2="time"}1' tempfile > $UTILLOG_TRAIN_FILE

# Fix utillog inference
tail -n +2 $UTILLOG_INF_FILE > tempfile
sed -i 's/|/ /g' tempfile
sed -i 's/  */ /g' tempfile
awk 'NR==1{$1="date"; $2="time"}1' tempfile > $UTILLOG_INF_FILE

# Cleanup
rm tempfile