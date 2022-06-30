export BERT_DIR=data/init_bert/uncased_L-12_H-768_A-12
export CONFIG_DIR=configs
export GLUE_DIR=data/glue
export SAVE_DIR=output

CUDA_VISIBLE_DEVICES=1 python run.py \
    --seed 1 \
    --output_dir $SAVE_DIR/camtl \
    --tasks all \
    --sample anneal \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/ \
    --vocab_file $BERT_DIR/vocab.txt \
    --bert_config_file $CONFIG_DIR/camtl_config.json \
    --init_checkpoint $BERT_DIR/pytorch_model.bin \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 25.0 \
    --gradient_accumulation_steps 1




  

  