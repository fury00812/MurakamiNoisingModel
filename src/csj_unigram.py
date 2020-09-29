"""
Create each sequence of morphemes, POS tags, pronunciations, and noise labels.
input: CSJ directory path
output: sequences.txt, input.(txt|pth), output.(txt|pth)
"""
import os
import argparse
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import pickle
import numpy as np

IGNOR_MARKER = ["感動詞","言いよどみ"]
STOP_MARKER = ["[文末]"]
SPLIT_MARKER = ["係助詞"]

filler_count = {}
error_count = {}

ext_ptn = "\(. (.*);.*\)" 
rm_ptn = "[()<>,.!?:;\-\" ]" 

def get_args():
    parser = argparse.ArgumentParser(description="csj_shaper.py")
    parser.add_argument("--in_dir", required="True")
    parser.add_argument("--out_dir", required="True")
    return parser.parse_args()


def get_files():
    assert os.path.exists(args.in_dir)
    DIR = Path(args.in_dir)
    file_paths = list(DIR.glob("*.xml"))
    return file_paths 


def token_clean(ext_ptn, rm_ptn, token):
    """
    clean a token
    :param ext_ptn: extraction pattern
    :param rm_path: to be removed characters pattern
    :param token: one word 
    """
    if token is None: 
        return "<UNK>"    
    match = re.match(ext_ptn, token)
    if match:
        token = match.group(1)
    token = re.sub(rm_ptn, "", token)
    return token


def make_label_seq(words):
    """
    make label sequence from word list that contains filler    
    :param words: a list of words 
    """
    labels = []
    f_words = []
    ptn = "\((F|D) (.*)\)"
    for i in range(len(words)):
        n = re.match(ptn,words[i])
        if n is not None:
            labels.append("<"+n.group(1)+">")
            filler = n.group(2)
            filler = token_clean(ext_ptn, rm_ptn, filler)
            f_words.append(filler)
            if n.group(1) is "F":
                filler_count[filler] = filler_count.get(filler, 0) + 1
            elif n.group(1) is "D":
                error_count[filler] = error_count.get(filler, 0) + 1
        else:
            labels.append("O")
            f_words.append(token_clean(ext_ptn, rm_ptn, words[i]))
        assert len(labels) == len(f_words)
    return labels, f_words


def parse_xml(xml_file):
    """
    Parse xml file
    :param xml_file: csj's xml file
    """
    tree = ET.parse(xml_file)
    talk = tree.getroot()
    
    sents = []
    labels = []

     
    f_words = []
    for ipu in talk: # sentence1, sentence2, ...
        for luw in ipu:
            boundary_label = None 
            luw_pos_info = luw.get("LUWMiscPOSInfo1") # SPLIT check
            for suw in luw:
                suw_token = suw.get("OrthographicTranscription") # 0:word
                suw_pos = suw.get("SUWPOS") # 1:POS
                boundary_label = suw.get("ClauseBoundaryLabel") # STOP check

                # (F *)や(D *)で囲われていない感動詞や言い淀みを除外
                if suw_pos in IGNOR_MARKER:
                    ptn = "\((F|D) .*\)"
                    if not re.match(ptn, suw_token):
                        continue

                if suw_token is not None:
                    filler_token = token_clean(ext_ptn, "", suw_token)
                    f_words.append(filler_token)
            if luw_pos_info in SPLIT_MARKER:
                f_words.append("、")
        if boundary_label in STOP_MARKER:
            f_words.append("。")
            label, f_words = make_label_seq(f_words)
            sents.append(f_words)
            labels.append(label)
            f_words = []
        elif boundary_label is not None:
            f_words.append("、")

    return sents, labels


def main():
    all_sents = []
    all_labels = []

    file_paths = get_files()
    for xml_file in file_paths:
        assert os.path.exists(xml_file)
        sents, labels = parse_xml(xml_file)
        all_sents += sents
        all_labels += labels

    # save tokens/labels
    out_path = Path(args.out_dir) / Path("csj.tok")
    with open(out_path, mode="w") as f:
        for sents in all_sents:
            f.write(" ".join(sents)+"\n")
    out_path = Path(args.out_dir) / Path("csj.label")
    with open(out_path, mode="w") as f:
        for labels in all_labels:
            f.write(" ".join(labels)+"\n")

    # save filler/error count
    out_path = Path(args.out_dir) / Path("fillers.csv")
    filler_l = []
    filler_prob_l = []
    with open(out_path, mode="w") as f:
        for word, count in sorted(filler_count.items(), key=lambda x: -x[1]):
            f.write("{},{}\n".format(word,count))
            filler_l.append(word)
            filler_prob_l.append(count)

    out_path = Path(args.out_dir) / Path("errors.csv")
    error_l = []
    error_prob_l = []
    with open(out_path, mode="w") as f:
        for word, count in sorted(error_count.items(), key=lambda x: -x[1]):
            f.write("{},{}\n".format(word,count))
            error_l.append(word)
            error_prob_l.append(count)

    # save probabilities
    out_path = Path(args.out_dir) / Path("probs.pkl")
    filler_prob_l = np.array(filler_prob_l, dtype=float)
    filler_prob_l = (filler_prob_l / np.sum(filler_prob_l)).tolist()
    error_prob_l = np.array(error_prob_l, dtype=float)
    error_prob_l = (error_prob_l / np.sum(error_prob_l)).tolist()
    assert len(filler_l) == len(filler_prob_l)
    assert len(error_l) == len(error_prob_l)
    probs_dict = {}
    probs_dict["<F>"] = [filler_l, filler_prob_l]
    probs_dict["<D>"] = [error_l, error_prob_l]
    with open(out_path, mode="wb") as f:
        pickle.dump(probs_dict, f)


if __name__ == "__main__":
    args = get_args()
    main()
