#!/bin/bash --login

#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --mem=50G  
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm-%A_%a.out

cd /mnt/ufs18/home-003/f0101170/DC/dance/
module purge
module load GCC/11.2.0  OpenMPI/4.1.1 GCCcore/11.2.0
module load Python/3.9.6
module load CUDA/11.4.2
module load R/4.1
source /mnt/ufs18/home-003/f0101170/sc_env_new/bin/activate
cd /mnt/ufs18/home-003/f0101170/DC/dance/


export BERT_DIR=data/init_bert/uncased_L-12_H-768_A-12
export CONFIG_DIR=configs
export GLUE_DIR=data/glue
export SAVE_DIR=output


python run.py \
  --seed 1 \
  --output_dir $SAVE_DIR/bert_n_pals \
  --tasks all \
  --sample 'anneal'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/ \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $CONFIG_DIR/bert_and_pals_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 40.0 \
  --gradient_accumulation_steps 1
