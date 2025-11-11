#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1  # Helps in debugging

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
root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=weather

random_seed=2021
train_epochs=50
patience=5
learning_rate=0.0001

# Define sequence and prediction lengths
seq_len_list=(720)
pred_len_list=(96 192 336 720)

# Create dataset-specific log folder
if [ ! -d "./logs/LongForecasting/${data_name}" ]; then
    mkdir ./logs/LongForecasting/${data_name}
fi

# ==============================
# Nested loops over seq_len and pred_len
# ==============================
for seq_len in "${seq_len_list[@]}"; do

  # Create a folder for this sequence length
  seq_log_dir="./logs/LongForecasting/${data_name}/${seq_len}"
  mkdir -p "$seq_log_dir"

  for pred_len in "${pred_len_list[@]}"; do
    echo "Running ${data_name} with seq_len=${seq_len}, pred_len=${pred_len} ..."

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
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 64 \
      --d_ff 64 \
      --dropout 0.5 \
      --fc_dropout 0.25 \
      --kernel_list 5 7 11 15 \
      --patch_len 48 32 16 6 3 \
      --des Exp \
      --lradj 'TST' \
      --train_epochs $train_epochs \
      --patience $patience \
      --top_k 5 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate $learning_rate \
      > "${seq_log_dir}/${model_name}_${data_name}_seq${seq_len}_pred${pred_len}.log"

    wait
  done
done
