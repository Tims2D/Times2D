export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=Times2D
root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2021

seq_len=720

#####################################pred_len=96 ######################################################
pred_len=96
#self.period_list[720 280 41]
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
  --train_epochs 25 \
  --patience 5 \
  --top_k 5 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log


#####################################pred_len=192 ######################################################
pred_len=192
#self.period_list[720, 360, 240, 144, 24, 12]

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
  --label_len 48\
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
  --train_epochs 10 \
  --patience 5 \
  --top_k 5 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log

#####################################pred_len=336 ######################################################
pred_len=336
#self.period_list = [720, 360, 110, 90, 48]
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
  --train_epochs 50 \
  --patience 5 \
  --top_k 5 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log


#####################################pred_len=720 ######################################################
pred_len=720
#self.period_list = [720, 360, 24] 
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
  --label_len 96\
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 32 \
  --dropout 0.6 \
  --fc_dropout 0.25 \
  --kernel_list 5 7 11 15 \
  --patch_len 48 32 16\
  --stride 48 32 16 6 3 \
  --des Exp \
  --lradj 'TST' \
  --train_epochs 15 \
  --patience 5 \
  --top_k 5 \
  --itr 1 --batch_size 128 --learning_rate 0.0001 > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log