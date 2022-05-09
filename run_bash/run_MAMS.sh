#!/bin/bash
exp_path=log/MAMS

if [ ! -d "$exp_path" ]; then
  echo "making new dir.."
  mkdir -p "$exp_path"
fi

DATE=$(date +%Y-%m-%d-%H_%M_%S)
CUDA_VISIBLE_DEVICES=0 python3 -u train.py \
	--data_dir data/V2/MAMS \
	--vocab_dir data/V2/MAMS \
	--batch_size 32 \
	--input_dropout 0.2 \
	--layer_dropout 0.1 \
	--attn_head 2 \
	--max_len 100 \
	--average_mapback \
    --plus_AA \
	--lr 3e-5 \
	--split_aa_graph \
	--aspect_graph_num_layer 1 \
	--con_dep_version con_dot_dep\
    --con_dep_conditional \
	--seed 2 \
	--num_epoch 20 2>&1 | tee $exp_path/training_$DATE.log