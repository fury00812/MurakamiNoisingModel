"""
Convert feature sequences from word to id for test data.
"""
import os
import pickle
import argparse
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader

from build_vocab import Dictionary


def get_args():
    parser = argparse.ArgumentParser(description="dataset.py")
    parser.add_argument("--data_pkl", required="True",
                        help="dictionaries of 3 types of sequences")
    parser.add_argument("--dict_pkl", required="True",
                        help="dictionaries of 3 types of Dictionary objects")
    parser.add_argument("--out_path", required="True")
    parser.add_argument("--max_vocab", type=int, default=16000,
                        help="maximum vocabulary size of words")
    parser.add_argument("--max_len", type=int, default=200,
                        help="maximum token of sequences")
    return parser.parse_args()


class DataSetGen:
    def __init__(self, seq_dicts, dict_dicts, max_vocab, max_len):
#        self.X = torch.LongTensor([[5,6],[7,8],[9,10],[11,12],[13,14]])
#        self.y = torch.LongTensor([0,1,2,3,4])
        self.seq_dicts = seq_dicts
        self.dict_dicts = dict_dicts
        self.max_vocab = max_vocab
        self.max_len = max_len


    def word2id(self, words, words_dict):
        ids = []
        for word in words:
             ids.append(words_dict.word2id.get(word, words_dict.unk_index))
        return torch.LongTensor(ids)


    def padding(self, ids, words_dict):
        ids = torch.cat([ids, torch.LongTensor([words_dict.pad_index\
                        for i in range(self.max_len - len(ids))])])
        return ids[:self.max_len]


    def __len__(self):
        return len(self.seq_dicts["words"])


    def __getitem__(self, index):
        seq_type = ["words", "pos", "pronuns"]

        dicts = {}
        for seq in seq_type:
            features = self.seq_dicts[seq][index]
            feature_dict = self.dict_dicts[seq]
            # word2id
            ids = self.word2id(features, feature_dict)
            # replace unknown words with <UNK>
            if seq == "words":
                ids.masked_fill_(ids>=self.max_vocab, feature_dict.unk_index)
                dicts["len"] = min(len(ids), self.max_len)
            # padding
            ids = self.padding(ids, feature_dict)
            dicts[seq] = ids

        return dicts


def main():
    assert os.path.exists(args.data_pkl)
    assert os.path.exists(args.dict_pkl)

    # load sequence dictionary
    with open(args.data_pkl, mode="rb") as f:
        seq_dicts = pickle.load(f)

    # load Dictionary dictionary
    with open(args.dict_pkl, mode="rb") as f:
        dict_dicts = pickle.load(f)

    # create DataSet
    dataset = DataSetGen(seq_dicts, dict_dicts, \
                                    args.max_vocab, args.max_len)

    # save id sequences
    torch.save(dataset, args.out_path)


if __name__ == "__main__":
    args = get_args()
    main()
