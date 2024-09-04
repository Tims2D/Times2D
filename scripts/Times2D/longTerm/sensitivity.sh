#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Create directories for logs if they don't exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/SensitivityAnalysis" ]; then
    mkdir ./logs/SensitivityAnalysis
fi

# Fixed parameters
model_name=Times2D
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
seq_len=720
random_seed=2021
pred_len=96
d_model=64
# Loop over d_model values
for seq_len in 2880
do
  python -u run.py \
    --task_name long_term_forecast \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}'_'${seq_len}'_'${pred_len}_dm${d_model} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 64\
    --d_ff 64 \
    --dropout 0.5 \
    --fc_dropout 0.25 \
    --patch_len 48 32 16 6 3 \
    --des Exp \
    --lradj 'TST' \
    --train_epochs 50 \
    --patience 5 \
    --top_k 5 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0001 >logs/SensitivityAnalysis/${model_name}_${model_id_name}_${seq_len}_${pred_len}_Seq${seq_len}.log
done

