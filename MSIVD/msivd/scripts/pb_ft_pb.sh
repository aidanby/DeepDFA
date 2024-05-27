if [ $# -lt 2 ]
then
echo fail
exit 1
fi

seed=$1
dataset=$2
shift
shift

python3 train.py \
  --model_name=pbfinetuned-pb_expl \
  --tb_dir=../tensorboard/ \
  --model_type=llama \
  --model_name_or_path=codellama/CodeLlama-13b-hf \
  --finetuned_path=../finetune_checkpoints/checkpoints_pb_expl/checkpoint \
  --output_dir=./saved_models \
  --do_train \
  --do_test \
  --train_data_file=../data/$dataset/train.csv \
  --eval_data_file=../data/$dataset/val.csv \
  --test_data_file=../data/$dataset/test.csv \
  --epochs 1 \
  --block_size 2048 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --learning_rate 1e-6 \
  --best_threshold 0.5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --use_finetuned_model \
  --seed $seed $@ 2>&1 | tee "train_${dataset}_expl.log"
