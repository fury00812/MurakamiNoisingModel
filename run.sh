#!/bin/bash
set -eu

# main paths
ROOT=$PWD
SRC_PATH=$ROOT/src
RAW_PATH=$ROOT/data/raw
SAVE_PATH=$ROOT/data/processed

# settings #:)
RAW_PATH=$RAW_PATH/CSJ
SAVE_PATH=$SAVE_PATH/CSJ
#RAW_PATH=$RAW_PATH/CSJ_test
#SAVE_PATH=$SAVE_PATH/CSJ_test
VOCAB_SIZE=16000

mkdir -p $SAVE_PATH

#
# Phase 0: Parse xml files and make source/target data
#
echo Phase 0: Preprocess
mkdir $SAVE_PATH/Phase0
echo -n running...
python $SRC_PATH/csj_shaper.py --in_dir $RAW_PATH --out_dir $SAVE_PATH/Phase0
echo done.

#
# Phase 1: Build Vocabulary
#
echo Phase 1: Build Vocabulary
mkdir -p $SAVE_PATH/Phase1
echo -n running...
python $SRC_PATH/build_vocab.py --data_pkl $SAVE_PATH/Phase0/sequences.pkl \
				--out_dir $SAVE_PATH/Phase1
echo done.

#
# Phase 2: Split train/valid/test data and convert sequences from word to id.
#
echo Phase 2: Create train/valid/test set
mkdir -p $SAVE_PATH/Phase2
echo -n running...
python $SRC_PATH/dataset.py \
	--data_pkl $SAVE_PATH/Phase0/sequences.pkl \
	--dict_pkl $SAVE_PATH/Phase1/dictionaries.pkl \
	--out_dir $SAVE_PATH/Phase2 --valid_size 1000 --test_size 1000 \
	--max_vocab $VOCAB_SIZE --max_len 100
echo done.

#
# Phase 3: Train the model
#
echo Phase 3: Train the model
EXP_PATH=$SAVE_PATH/Phase3/no-tmp_batch100 #:)
mkdir -p $EXP_PATH
echo running...
export CUDA_VISIBLE_DEVICES=0
python $SRC_PATH/main.py \
	--train_dataset $SAVE_PATH/Phase2/trainset.pth \
	--valid_dataset $SAVE_PATH/Phase2/validset.pth \
	--test_dataset $SAVE_PATH/Phase2/testset.pth \
	--dicts $SAVE_PATH/Phase1/dictionaries.pkl \
	--out_dir $EXP_PATH --num_epoch 1000 --valid_epoch 1 \
	--early_stopping 20 --batch_size 100 --max_vocab $VOCAB_SIZE \
	--words_dim 200 --pos_dim 50 --pros_dim 50 --hidden_dim 200 \
	--num_layers 2 --learning_rate 0.1 --dropout 0.2 \
	--temperature 0.15 | tee $EXP_PATH/train.log
#	--eval_only --save_path "best_epoch.pth"
echo done.

#
# Phase 4: Calculate unigram probabilities of filler/errors
#
echo Phase 4: Calculate unigram probs
mkdir -p $SAVE_PATH/Phase4
echo -n running...
python $SRC_PATH/csj_unigram.py --in_dir $RAW_PATH --out_dir $SAVE_PATH/Phase4
echo done.
