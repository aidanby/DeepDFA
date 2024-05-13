if [ $# -lt 2 ]
then
echo fail
exit 1
fi

seed=$1
dataset=$2
shift
shift

python3 linevul_main.py \
  --model_name=finetuned-bigvul-nodfa \
  --tb_dir=../tensorboard/finetuned-bigvul-nodfa/ \
  --finetuned_path=../finetune_checkpoints/checkpoints_bigvul_codellama/step_2500 \
  --model_type=llama \
  --tokenizer_name=codellama/CodeLlama-13b-Instruct-hf \
  --model_name_or_path=codellama/CodeLlama-13b-Instruct-hf \
  --output_dir=./saved_models \
  --do_train \
  --do_test \
  --train_data_file=../data/$dataset/train.csv \
  --eval_data_file=../data/$dataset/val.csv \
  --test_data_file=../data/$dataset/test.csv \
  --epochs 5 \
  --block_size 512 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --use_finetuned_model \
  --no_flowgnn \
  --really_no_flowgnn \
  --seed $seed $@ 2>&1 | tee "train_${dataset}_${seed}.log"