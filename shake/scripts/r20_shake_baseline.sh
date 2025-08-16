#!/bin/bash

# ResNet-20 SHAKE Baseline Training (Standard Knowledge Distillation)
# This script trains ResNet-20 using standard SHAKE knowledge distillation

# Configuration
python_path="/home/yanhongwei/miniconda3/envs/DGIL/bin/python"
seeds=(0 1 2)
trials=(10 11 12)  # Use different trial numbers to avoid conflicts
gpus=(5 6 7)

# Training parameters
EPOCHS=240
BATCH_SIZE=64
LEARNING_RATE=0.05
WEIGHT_DECAY=5e-4
MOMENTUM=0.9
LR_DECAY_EPOCHS="150,180,210"
LR_DECAY_RATE=0.1

# SHAKE specific parameters
TEACHER_PATH="./save/models/resnet56_vanilla/ckpt_epoch_240.pth"
STUDENT_MODEL="resnet20"
DISTILL_METHOD="shake"
ALPHA=0      # weight for KL divergence loss
BETA=1       # weight for SHAKE loss
GAMMA=1      # weight for classification loss

echo "Starting ResNet-20 SHAKE Baseline Training"
echo "=========================================="
echo "Training parameters:"
echo "  Teacher: resnet56 (from $TEACHER_PATH)"
echo "  Student: $STUDENT_MODEL"
echo "  Distillation: $DISTILL_METHOD"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Loss weights: alpha=$ALPHA, beta=$BETA, gamma=$GAMMA"
echo "  Seeds: ${seeds[@]}"
echo "  Trials: ${trials[@]}"
echo "  GPUs: ${gpus[@]}"
echo ""

# Check if teacher model exists
if [ ! -f "$TEACHER_PATH" ]; then
    echo "Error: Teacher model not found at $TEACHER_PATH"
    echo "Please run: bash scripts/fetch_pretrained_teachers.sh"
    exit 1
fi

# Run experiments
for i in {0..2}; do
    seed=${seeds[$i]}
    trial=${trials[$i]}
    gpu=${gpus[$i]}
    exp_name="r20_shake_baseline_seed${seed}"

    echo "Starting $exp_name on GPU $gpu (trial=${trial})"

    screen -dmS "$exp_name" bash -c "
        cd /home/yanhongwei/spaced_kd/online_kd/SHAKE/shake && \
        CUDA_VISIBLE_DEVICES=$gpu \
        $python_path -W ignore train_student.py \
            --path_t $TEACHER_PATH \
            --distill $DISTILL_METHOD \
            --model_s $STUDENT_MODEL \
            --dataset cifar100 \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --lr_decay_epochs $LR_DECAY_EPOCHS \
            --lr_decay_rate $LR_DECAY_RATE \
            --weight_decay $WEIGHT_DECAY \
            --momentum $MOMENTUM \
            -a $ALPHA \
            -b $BETA \
            -r $GAMMA \
            --trial $trial \
            --print_freq 100 \
            --save_freq 40 \
            2>&1 | tee ./save/logs/r20_shake_baseline_seed${seed}.log
    "
done

echo ""
echo "All experiments started. Use 'screen -ls' to list sessions."
echo "Logs will be saved to: ./save/logs/r20_shake_baseline_seed{0,1,2}.log"
echo "Models will be saved to: ./save/models/"
echo ""
echo "To monitor progress:"
echo "  screen -r r20_shake_baseline_seed0  # Attach to seed 0"
echo "  screen -r r20_shake_baseline_seed1  # Attach to seed 1" 
echo "  screen -r r20_shake_baseline_seed2  # Attach to seed 2"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
