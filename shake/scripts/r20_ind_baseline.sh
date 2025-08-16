#!/bin/bash

# ResNet-20 Independent Baseline Training (No Knowledge Distillation)
# This script trains ResNet-20 independently without any distillation loss

# Configuration
python_path="/home/yanhongwei/miniconda3/envs/DGIL/bin/python"
seeds=(0 1 2)
gpus=(5 6 7)

# Training parameters
EPOCHS=240
BATCH_SIZE=64
LEARNING_RATE=0.05
WEIGHT_DECAY=5e-4
MOMENTUM=0.9
LR_DECAY_EPOCHS="150,180,210"
LR_DECAY_RATE=0.1

echo "Starting ResNet-20 Independent Baseline Training"
echo "================================================"
echo "Training parameters:"
echo "  Model: resnet20"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Seeds: ${seeds[@]}"
echo "  GPUs: ${gpus[@]}"
echo ""

# Run experiments
for i in {0..2}; do
    seed=${seeds[$i]}
    gpu=${gpus[$i]}
    exp_name="r20_ind_baseline_seed${seed}"

    echo "Starting $exp_name on GPU $gpu"

    screen -dmS "$exp_name" bash -c "
        cd /home/yanhongwei/spaced_kd/online_kd/SHAKE/shake && \
        CUDA_VISIBLE_DEVICES=$gpu \
        $python_path -W ignore train_teacher.py \
            --model resnet20 \
            --dataset cifar100 \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --lr_decay_epochs $LR_DECAY_EPOCHS \
            --lr_decay_rate $LR_DECAY_RATE \
            --weight_decay $WEIGHT_DECAY \
            --momentum $MOMENTUM \
            --trial $seed \
            --print_freq 100 \
            --save_freq 40 \
            2>&1 | tee ./save/logs/r20_ind_baseline_seed${seed}.log
    "
done

echo ""
echo "All experiments started. Use 'screen -ls' to list sessions."
echo "Logs will be saved to: ./save/logs/r20_ind_baseline_seed{0,1,2}.log"
echo "Models will be saved to: ./save/models/"
echo ""
echo "To monitor progress:"
echo "  screen -r r20_ind_baseline_seed0  # Attach to seed 0"
echo "  screen -r r20_ind_baseline_seed1  # Attach to seed 1" 
echo "  screen -r r20_ind_baseline_seed2  # Attach to seed 2"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
