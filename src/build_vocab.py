"""
Build vocabulary
input: sequences.pkl 
output: dictionaries.pkl, (*_count.txt)
"""
import os
import argparse
import re
import pickle
from pathlib import Path

BOS_WORD = "<BOS>"
EOS_WORD = "<EOS>"
PAD_WORD = "<PAD>"
UNK_WORD = "<UNK>"

NUM_SPECIAL_WORDS=4
BOS=0
EOS=1
PAD=2
UNK=3

out_file = "dictionaries.pkl"

def get_args():
    parser = argparse.ArgumentParser(description="build_vocab.py")
    parser.add_argument("--data_pkl", required="True",
                        help="dictionaries of 4 types of sequences")
    parser.add_argument("--out_dir", required="True")
    return parser.parse_args()


class Dictionary(object):
    def __init__(self):
        self.id2word = {}
        self.word2id = {
            BOS_WORD: BOS,
            EOS_WORD: EOS,
            PAD_WORD: PAD,
            UNK_WORD: UNK
        }
        self.unk_index = UNK
        self.pad_index = PAD
        self.num_words = 0


#    def prune(self, max_vocab):
#        """
#        Limit the vocabulary size.
#        """
#        assert max_vocab >= 1, max_vocab
#        self.id2word = {k: v for k, v in self.id2word.items() \
#                        if k < max_vocab}
#        self.word2id = {v: k for k, v in self.id2word.items()}


    def build_vocab(self, sequences):
        """
        count num of appearances and build vocabulary
        :param sequences: a list of sequences
        """
        word_count = {}
        word_descent = {}
        for words in sequences:
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
        for word, count in sorted(word_count.items(), key=lambda x: -x[1]):
            word_descent[word] = count
        for s in [BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD]:
            if s in word_descent:
                del word_descent[s]
        for i, (word,count) in enumerate(word_descent.items()):
            self.word2id[word] = NUM_SPECIAL_WORDS + i
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.num_words = len(self.word2id)
        return word_descent


def main():
    out_path = Path(args.out_dir) / Path(out_file)

    seq_type = ["words", "pos", "pronuns", "labels"]

    with open(args.data_pkl, mode="rb") as f:
        seq_dicts = pickle.load(f)


    # Create Dictionary objects of each feature
    dicts = {}
    for seq in seq_type:
        feature_dict = Dictionary()
        feature_count = feature_dict.build_vocab(seq_dicts[seq])
        csv_path = Path(args.out_dir) / Path("{}_count.csv".format(seq))
        with open(csv_path, mode="w") as f:
            for word, count in sorted(feature_count.items(), key=lambda x: -x[1]):
                f.write("{},{}\n".format(str(word), str(count)))
        dicts[seq] = feature_dict

    # Save the dictionary of Dictionary objects
    with open(out_path, mode="wb") as f:
        pickle.dump(dicts, f)


if __name__ == "__main__":
    args = get_args()
    main() 
