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

# ==============================
# Base configuration
# ==============================
model_name=Times2D
root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

# Create dataset-specific log folder
if [ ! -d "./logs/LongForecasting/${data_name}" ]; then
    mkdir ./logs/LongForecasting/${data_name}
fi

random_seed=2021
train_epochs=50
seq_len=720

# ==============================
# pred_len = 96
# ==============================
pred_len=96
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
  --dropout 0.6 \
  --fc_dropout 0.25 \
  --kernel_list 5 7 11 15 \
  --patch_len 48 32 16 6 3 \
  --stride 48 32 16 6 3 \
  --des Exp \
  --lradj 'TST' \
  --train_epochs $train_epochs \
  --patience 5 \
  --top_k 5 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  > logs/LongForecasting/${data_name}/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log

# ==============================
# pred_len = 192
# ==============================
pred_len=192
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
  --label_len 48 \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 128 \
  --dropout 0.6 \
  --fc_dropout 0.25 \
  --kernel_list 5 7 11 15 \
  --patch_len 48 32 16 6 3 1 \
  --stride 48 32 16 6 3 \
  --des Exp \
  --lradj 'TST' \
  --train_epochs $train_epochs \
  --patience 5 \
  --top_k 5 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  > logs/LongForecasting/${data_name}/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log

# ==============================
# pred_len = 336
# ==============================
pred_len=336
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
  --patch_len 48 32 16 6 3 \
  --stride 48 32 16 6 3 \
  --des Exp \
  --lradj 'TST' \
  --train_epochs $train_epochs \
  --patience 5 \
  --top_k 5 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  > logs/LongForecasting/${data_name}/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log

# ==============================
# pred_len = 720
# ==============================
pred_len=720
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
  --label_len 96 \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 32 \
  --dropout 0.6 \
  --fc_dropout 0.25 \
  --kernel_list 5 7 11 15 \
  --patch_len 48 32 16 6 3 \
  --stride 48 32 16 6 3 \
  --des Exp \
  --lradj 'TST' \
  --train_epochs $train_epochs \
  --patience 5 \
  --top_k 5 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 \
  > logs/LongForecasting/${data_name}/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
