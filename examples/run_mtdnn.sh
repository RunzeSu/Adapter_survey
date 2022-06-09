#!/bin/bash --login

#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --mem=50G  
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm-%A_%a.out

cd /mnt/ufs18/home-052/surunze/eg_multitask/
source eg_env/bin/activate
module purge
module load GCCcore/8.3.0
module load Python/3.8.3
cd /mnt/ufs18/home-052/surunze/Gated_bert/

export BERT_DIR=data/init_bert/uncased_L-12_H-768_A-12
export CONFIG_DIR=data/configs
export GLUE_DIR=data/glue
export SAVE_DIR=output


python run_multitask.py \
  --seed 1 \
  --output_dir $SAVE_DIR/mtdnn \
  --tasks all \
  --sample 'anneal'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/ \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $CONFIG_DIR/embert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 40.0 \
  --gradient_accumulation_steps 1
