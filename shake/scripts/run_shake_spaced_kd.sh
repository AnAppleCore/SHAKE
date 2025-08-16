#!/bin/bash

# Script for running SHAKE with Spaced Knowledge Distillation
# This script demonstrates different space_interval configurations

echo "SHAKE Spaced Knowledge Distillation Training Scripts"
echo "===================================================="

# Configuration
PYTHON_PATH="/home/yanhongwei/miniconda3/envs/DGIL/bin/python"
TEACHER_PATH="./save/models/resnet56_vanilla/ckpt_epoch_240.pth"
STUDENT_MODEL="resnet20"
DATASET="cifar100"
EPOCHS=480
BATCH_SIZE=64
LEARNING_RATE=0.05

# Check if teacher model exists
if [ ! -f "$TEACHER_PATH" ]; then
    echo "Error: Teacher model not found at $TEACHER_PATH"
    echo "Please run: bash scripts/fetch_pretrained_teachers.sh"
    exit 1
fi

echo "Available training modes:"
echo "1. Standard SHAKE (no spaced KD)"
echo "2. SHAKE with Spaced KD (interval=1.0)"
echo "3. SHAKE with Spaced KD (interval=1.5)"
echo "4. SHAKE with Spaced KD (interval=2.0)"
echo "5. SHAKE with Spaced KD (interval=3.0)"
echo ""

# Function to run training
run_training() {
    local space_interval=$1
    local trial=$2
    local description=$3
    
    echo "Starting: $description"
    echo "Space interval: $space_interval"
    echo "Trial: $trial"
    echo "Command:"
    
    if [ "$space_interval" == "-1" ]; then
        # Standard SHAKE without spaced KD
        echo "CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH train_student.py \\"
        echo "    --path_t $TEACHER_PATH \\"
        echo "    --distill shake \\"
        echo "    --model_s $STUDENT_MODEL \\"
        echo "    --dataset $DATASET \\"
        echo "    --epochs $EPOCHS \\"
        echo "    --batch_size $BATCH_SIZE \\"
        echo "    --learning_rate $LEARNING_RATE \\"
        echo "    -a 0 -b 1 \\"
        echo "    --trial $trial"
        echo ""
        
        # Uncomment the following line to actually run the training
        # CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH train_student.py \
        #     --path_t $TEACHER_PATH \
        #     --distill shake \
        #     --model_s $STUDENT_MODEL \
        #     --dataset $DATASET \
        #     --epochs $EPOCHS \
        #     --batch_size $BATCH_SIZE \
        #     --learning_rate $LEARNING_RATE \
        #     -a 0 -b 1 \
        #     --trial $trial
    else
        # SHAKE with spaced KD
        echo "CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH train_student.py \\"
        echo "    --path_t $TEACHER_PATH \\"
        echo "    --distill shake \\"
        echo "    --model_s $STUDENT_MODEL \\"
        echo "    --dataset $DATASET \\"
        echo "    --epochs $EPOCHS \\"
        echo "    --batch_size $BATCH_SIZE \\"
        echo "    --learning_rate $LEARNING_RATE \\"
        echo "    -a 0 -b 1 \\"
        echo "    --space_interval $space_interval \\"
        echo "    --trial $trial"
        echo ""
        
        # Uncomment the following line to actually run the training
        # CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH train_student.py \
        #     --path_t $TEACHER_PATH \
        #     --distill shake \
        #     --model_s $STUDENT_MODEL \
        #     --dataset $DATASET \
        #     --epochs $EPOCHS \
        #     --batch_size $BATCH_SIZE \
        #     --learning_rate $LEARNING_RATE \
        #     -a 0 -b 1 \
        #     --space_interval $space_interval \
        #     --trial $trial
    fi
}

# Example training configurations
echo "Example training commands (uncomment in script to run):"
echo ""

# 1. Standard SHAKE
run_training "-1" "0" "Standard SHAKE (baseline)"

# 2. SHAKE with different spaced KD intervals
run_training "1.0" "1" "SHAKE with Spaced KD (interval=1.0)"
run_training "1.5" "2" "SHAKE with Spaced KD (interval=1.5)"
run_training "2.0" "3" "SHAKE with Spaced KD (interval=2.0)"
run_training "3.0" "4" "SHAKE with Spaced KD (interval=3.0)"

echo ""
echo "To run actual training:"
echo "1. Edit this script and uncomment the desired training command"
echo "2. Or run the commands manually"
echo ""
echo "Note: Make sure you have:"
echo "- Downloaded the teacher models: bash scripts/fetch_pretrained_teachers.sh"
echo "- Sufficient GPU memory available"
echo "- CIFAR-100 dataset will be downloaded automatically"
