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
task_name=long_term_forecast
root_path_name=./dataset/
data_path_name=exchange_rate.csv
data_name=exchange_rate
model_id_name=${data_name}
random_seed=2024
label_len=48
learning_rate=0.0001

# Create dataset-specific log folder
if [ ! -d "./logs/LongForecasting/${data_name}" ]; then
    mkdir ./logs/LongForecasting/${data_name}
fi

# ==============================
# Experiment setup
# ==============================
seq_len_list=(96)
pred_len_list=(96 192 336 720)

# ==============================
# Loop over sequence & pred lengths
# ==============================
for seq_len in "${seq_len_list[@]}"; do

  # Create folder for this sequence length
  seq_log_dir="./logs/LongForecasting/${data_name}/${seq_len}"
  mkdir -p "$seq_log_dir"

  for pred_len in "${pred_len_list[@]}"; do

    echo "Running ${data_name} with seq_len=${seq_len}, pred_len=${pred_len} ..."

    # Run experiment and save log inside seq_len folder
    python -u run.py \
      --random_seed $random_seed \
      --task_name $task_name \
      --model_id ${model_name}_${data_name}_seq${seq_len}_pred${pred_len} \
      --is_training 1 \
      --model $model_name \
      --data $data_name \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 64 \
      --d_ff 64 \
      --dropout 0.5 \
      --fc_dropout 0.25 \
      --patch_len 48 32 16 6 3 \
      --des Exp \
      --lradj 'TST' \
      --train_epochs 50 \
      --patience 5 \
      --top_k 5 \
      --batch_size 64 \
      --learning_rate $learning_rate \
      --itr 1 \
      > "${seq_log_dir}/${model_name}_${data_name}_seq${seq_len}_pred${pred_len}.log"

    wait
  done
done
