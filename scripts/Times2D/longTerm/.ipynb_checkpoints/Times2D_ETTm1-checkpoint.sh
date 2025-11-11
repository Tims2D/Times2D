#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==============================
# Create log folders if missing
# ==============================
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=Times2D
root_path_name=./dataset/ETT-small/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

# Create dataset-specific log folder
if [ ! -d "./logs/LongForecasting/${data_name}" ]; then
    mkdir ./logs/LongForecasting/${data_name}
fi

random_seed=2021
seq_len=720
train_epochs=50
patience=5

# ==============================
# Loop over all pred_len values
# ==============================
for pred_len in 96 192 336 720
do
  echo "Running ${data_name} with pred_len=${pred_len} ..."
  
  python -u run.py \
    --task_name long_term_forecast \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_${seq_len}_${pred_len} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 64 \
    --d_ff 64 \
    --dropout 0.5 \
    --fc_dropout 0.25 \
    --kernel_list 5 7 11 15 \
    --period 48 90 110 360 720 \
    --patch_len 48 32 16 6 3 \
    --stride 48 32 16 6 3 \
    --des Exp \
    --lradj 'TST' \
    --train_epochs $train_epochs \
    --patience $patience \
    --top_k 5 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    > logs/LongForecasting/${data_name}/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
done
