"""
Create each sequence of morphemes, POS tags, pronunciations, and noise labels for test data.
input: raw file (e.g. filename=aspec.train.ja)
output: filename.features, filename.pkl
"""
import os
import argparse
import re
from pathlib import Path
import pickle
import MeCab


def get_args():
    parser = argparse.ArgumentParser(description="extract_features.py")
    parser.add_argument("--raw_path", required="True")
    parser.add_argument("--out_dir", required="True")
    return parser.parse_args()


def get_features(tagger, text):
    word_l = []
    pos_l = []
    pros_l = []
    node = tagger.parseToNode(text)
    while node:
        word = node.surface
        pos = node.feature.split(",")[0]
        pros = node.feature.split(",")[-1]

        if pos=="記号" and word=="，":
            pos = "読点"
            word = "、"
        elif pos=="記号" and word=="。":
            pos = "句点"

        word_l.append(word)
        pos_l.append(pos)
        pros_l.append(pros)
        node = node.next

    word_l[0] = "<BOS>"
    pos_l[0] = "<BOS>"
    pros_l[0] = "<BOS>"

    word_l[-1] = "<EOS>"
    pos_l[-1] = "<EOS>"
    pros_l[-1] = "<EOS>"

    return (" ").join(word_l), word_l, pos_l, pros_l

def main():
    all_sents = []
    all_words = []
    all_pos = []
    all_pros = []

    with open(args.raw_path, mode="r") as f:
        lines = f.readlines()

    tagger = MeCab.Tagger("")
    tagger.parse("")
    for sent in lines:
        sents, words, pos, pros = get_features(tagger, sent)
        all_sents.append(sents)
        all_words.append(words)
        all_pos.append(pos)
        all_pros.append(pros)

    # save data
    name = os.path.basename(args.raw_path)
    out_path = Path(args.out_dir) / Path("{}.feats.txt".format(name))
    with open(out_path, mode="w") as f:
        for i in range(len(all_sents)):
            f.write("SENT {}\n".format(i))
            f.write("w:{}\n".format(all_words[i]))
            f.write("p:{}\n".format(all_pos[i]))
            f.write("r:{}\n".format(all_pros[i]))


    out_path = Path(args.out_dir) / Path("{}.feats.pkl".format(name))
    data = {
        "words": all_words,
        "pos": all_pos,
        "pronuns": all_pros
    }
    with open(out_path, mode="wb") as f:
        pickle.dump(data, f) 


if __name__ == "__main__":
    args = get_args()
    main()
