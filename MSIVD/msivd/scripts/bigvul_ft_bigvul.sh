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
  --model_name=finetuned-bigvul-test \
  --tb_dir=../tensorboard/ \
  --finetuned_path=../finetune_checkpoints/checkpoints_bigvul_expl/checkpoint \
  --model_type=llama \
  --model_name_or_path=codellama/CodeLlama-7b-hf \
  --output_dir=./saved_models \
  --do_train \
  --do_test \
  --train_data_file=../data/$dataset/train.csv \
  --eval_data_file=../data/$dataset/val.csv \
  --test_data_file=../data/$dataset/test.csv \
  --epochs 5 \
  --block_size 256 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --learning_rate 1e-4 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed $seed $@ 2>&1 | tee "train_${dataset}_expl.log"
