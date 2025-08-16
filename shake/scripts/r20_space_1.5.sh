#!/bin/bash

# ResNet-20 SHAKE with Spaced Knowledge Distillation (space_interval=1.5)
# This script trains ResNet-20 using SHAKE with spaced KD, running for double epochs

# Configuration
python_path="/home/yanhongwei/miniconda3/envs/DGIL/bin/python"
seeds=(0 1 2)
trials=(20 21 22)  # Use different trial numbers to avoid conflicts
gpus=(5 6 7)

# Training parameters (double epochs for spaced KD experiment)
EPOCHS=480  # Double the standard SHAKE epochs (240 * 2)
BATCH_SIZE=64
LEARNING_RATE=0.05
WEIGHT_DECAY=5e-4
MOMENTUM=0.9
LR_DECAY_EPOCHS="300,360,420"  # Adjusted for double epochs
LR_DECAY_RATE=0.1

# SHAKE specific parameters
TEACHER_PATH="./save/models/resnet56_vanilla/ckpt_epoch_240.pth"
STUDENT_MODEL="resnet20"
DISTILL_METHOD="shake"
ALPHA=0      # weight for KL divergence loss
BETA=1       # weight for SHAKE loss
GAMMA=1      # weight for classification loss
SPACE_INTERVAL=1.5  # Spaced KD interval

echo "Starting ResNet-20 SHAKE with Spaced KD Training"
echo "==============================================="
echo "Training parameters:"
echo "  Teacher: resnet56 (from $TEACHER_PATH)"
echo "  Student: $STUDENT_MODEL"
echo "  Distillation: $DISTILL_METHOD"
echo "  Space interval: $SPACE_INTERVAL epochs"
echo "  Epochs: $EPOCHS (double standard)"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  LR decay epochs: $LR_DECAY_EPOCHS"
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
    exp_name="r20_space_1.5_seed${seed}"

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
            --space_interval $SPACE_INTERVAL \
            --trial $trial \
            --print_freq 100 \
            --save_freq 40 \
            2>&1 | tee ./save/logs/r20_space_1.5_seed${seed}.log
    "
done

echo ""
echo "All experiments started. Use 'screen -ls' to list sessions."
echo "Logs will be saved to: ./save/logs/r20_space_1.5_seed{0,1,2}.log"
echo "Models will be saved to: ./save/models/"
echo ""
echo "Expected training behavior:"
echo "  - First 1.5 epochs: KD loss disabled"
echo "  - Next 1.5 epochs: KD loss enabled"
echo "  - Pattern repeats throughout training"
echo ""
echo "To monitor progress:"
echo "  screen -r r20_space_1.5_seed0  # Attach to seed 0"
echo "  screen -r r20_space_1.5_seed1  # Attach to seed 1" 
echo "  screen -r r20_space_1.5_seed2  # Attach to seed 2"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
