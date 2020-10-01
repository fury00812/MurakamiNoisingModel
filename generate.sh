#!/bin/bash
set -eu

# main paths
ROOT=$PWD
SRC_PATH=$ROOT/src
RAW_PATH=$ROOT/data/raw
SAVE_PATH=$ROOT/data/processed

# settings #:)
VOCAB_SIZE=16000
CSJ_PATH=$SAVE_PATH/CSJ
RAW_PATH=$RAW_PATH/ASPEC
SAVE_PATH=$SAVE_PATH/ASPEC

# files full pahts #:)
BASE_NAME=aspec.test.ja
RAW_PATH=$RAW_PATH/$BASE_NAME
DICT_PATH=$CSJ_PATH/Phase1/dictionaries.pkl
MODEL_PATH=$CSJ_PATH/Phase3/first/best_epoch.pth
NOISE_MODEL_PATH=$CSJ_PATH/Phase4/probs.pkl

mkdir -p $SAVE_PATH

#
# Phase 0: pack feature(word/pos/pronunciations) sequences from raw text
#
echo Phase 0: Preprocess
mkdir -p $SAVE_PATH/Phase0
echo -n running...
python $SRC_PATH/extract_features.py --raw_path $RAW_PATH --out_dir $SAVE_PATH/Phase0
echo done.

#
# Phase 1: Convert feature sequences from word to id.
#
echo Phase 1: Create train/valid/test set
mkdir -p $SAVE_PATH/Phase1
echo -n running...
python $SRC_PATH/dataset_gen.py \
	--data_pkl $SAVE_PATH/Phase0/$BASE_NAME.feats.pkl \
	--dict_pkl $DICT_PATH \
	--out_path $SAVE_PATH/Phase1/$BASE_NAME.pth \
	--max_vocab $VOCAB_SIZE --max_len 100
echo done.

#
# Phase 2: Generate label sequences by pretrained model
#
echo Phase 2: Train the model
mkdir -p $SAVE_PATH/Phase2
echo running...
export CUDA_VISIBLE_DEVICES=0
python $SRC_PATH/main.py \
	--test_dataset $SAVE_PATH/Phase1/$BASE_NAME.pth \
	--dicts $DICT_PATH --save_path $MODEL_PATH \
	--out_path $SAVE_PATH/Phase2/$BASE_NAME \
	--batch_size 20 --max_vocab $VOCAB_SIZE \
	--words_dim 200 --pos_dim 50 --pros_dim 50 --hidden_dim 200 \
	--num_layers 2 --learning_rate 0.1 --dropout 0.2 \
	--temperature 0.15 --generate
echo done.
#
##
## Phase 3: Calculate unigram probabilities of filler/errors
##
#echo Phase 4: Calculate unigram probs
#mkdir -p $SAVE_PATH/Phase4
#echo -n running...
#python $SRC_PATH/csj_unigram.py --in_dir $RAW_PATH --out_dir $SAVE_PATH/Phase4
#echo done.
