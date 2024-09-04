export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=Times2D
root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2021

seq_len=720

for pred_len in 96 192 336 720  
do
  python -u run.py \
    --task_name long_term_forecast \
    --random_seed 2021 \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --model_id ETTm1'_'$seq_len'_'$pred_len \
    --model Times2D \
    --data ETTm1 \
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
    --patch_len 48 32 16  6 3 \
    --stride 48 32 16 6 3 \
    --des Exp \
    --lradj 'TST' \
    --train_epochs 50 \
    --patience 5 \
    --top_k 5 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done