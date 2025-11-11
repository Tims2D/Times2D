export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ShortTermForecasting" ]; then
    mkdir ./logs/ShortTermForecasting
fi

model_name=Times2D
random_seed=2021

##########################################################Yearly
#self.period_list = [720, 360, 110, 96, 48]  # M4 yearly
model_id=m4_Yearly
python -u run.py \
  --task_name short_term_forecast \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns Yearly \
  --model_id $model_id \
  --model Times2D \
  --data m4 \
  --features M \
  --enc_in 1 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 32 \
  --d_ff 512 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --period 12 6 3 \
  --patch_len 48 32 16 6 3\
  --des Exp \
  --lradj 'TST' \
  --train_epochs 100 \
  --patience 10 \
  --top_k 5 \
  --loss 'SMAPE' \
  --itr 1 --batch_size 128 --learning_rate 0.0001 > logs/ShortTermForecasting/${model_name}_${model_id}_${seq_len}.log

#####################################################Quarterly
model_id=m4_Quarterly
python -u run.py \
  --task_name short_term_forecast \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns Quarterly \
  --model_id $model_id \
  --model Times2D \
  --data m4 \
  --features M \
  --enc_in 1 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_len 48 32 16 6 3 \
  --des Exp \
  --lradj 'TST' \
  --train_epochs 100 \
  --patience 10 \
  --top_k 5 \
  --loss 'SMAPE' \
  --itr 1 --batch_size 128 --learning_rate 0.0001 > logs/ShortTermForecasting/${model_name}_${model_id}_${seq_len}.log

#####################################################################Monthly
model_id=m4_Monthly
python -u run.py \
  --task_name short_term_forecast \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path ./dataset/m4/ \
  --seasonal_patterns Monthly \
  --model_id $model_id \
  --model Times2D \
  --data m4 \
  --features M \
  --enc_in 1 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_ff 256 \
  --dropout 0.5 \
  --fc_dropout 0.25 \
  --patch_len 48 32 16 6 3 \
  --des Exp \
  --lradj 'TST' \
  --train_epochs 100 \
  --patience 10 \
  --top_k 5 \
  --loss 'SMAPE' \
  --itr 1 --batch_size 128 --learning_rate 0.0001 > logs/ShortTermForecasting/${model_name}_${model_id}_${seq_len}.log