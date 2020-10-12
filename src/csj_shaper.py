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

IGNOR_MARKER = ["感動詞","言いよどみ"]
STOP_MARKER = ["[文末]"]
SPLIT_MARKER = ["係助詞"]

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
    :param ext_ptn: extractin pattern
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
    ptn = "\((F|D) .*\)"
    for i in range(len(words)-1):
        if not re.match(ptn,words[i]):
            n = re.match(ptn,words[i+1])
            if n is not None:
                labels.append("<"+n.group(1)+">")
            else:
                labels.append("O")
    labels.append("O")    
    return labels


def parse_xml(xml_file):
    """
    Parse xml file
    :param xml_file: csj"s xml file
    """
    tree = ET.parse(xml_file)
    talk = tree.getroot()
    
    sents = []
    seq_words = []
    seq_pos = []
    seq_pros = []
    seq_labels = []
    sent = ""    
    f_words = ["<BOS>"]
    words = ["<BOS>"] 
    pos = ["<BOS>"] 
    pros = ["<BOS>"]
    ext_ptn = "\(. (.*);.*\)" 
    rm_ptn = "[()<>,.!?:;\-\" ]" 
     
    for ipu in talk: # sentence1, sentence2, ...
        for luw in ipu:
            boundary_label = None 
            luw_pos_info = luw.get("LUWMiscPOSInfo1") # SPLIT check
            for suw in luw:
                suw_token = suw.get("OrthographicTranscription") # 0:word
                suw_pos = suw.get("SUWPOS") # 1:POS
                suw_pronun = suw.get("PhoneticTranscription") # 2:speech
                boundary_label = suw.get("ClauseBoundaryLabel") # STOP check

                # (F *)や(D *)で囲われていない感動詞や言い淀みを除外
                if suw_pos in IGNOR_MARKER:
                    ptn = "\((F|D) .*\)"
                    if not re.match(ptn, suw_token):
                        continue

                if suw_token is not None:
                    filler_token = token_clean(ext_ptn, "", suw_token)
                    suw_token = token_clean(ext_ptn, rm_ptn, suw_token)
                    suw_pos = token_clean(ext_ptn, rm_ptn, suw_pos)
                    suw_pronun = token_clean(ext_ptn, 
                                    re.sub("]","A-Za-z0-9]",rm_ptn), suw_pronun)
                    sent += filler_token
                    f_words.append(filler_token)
                    if suw_pos not in IGNOR_MARKER:
                        words.append(suw_token) 
                        pos.append(suw_pos)
                        pros.append(suw_pronun)
            if luw_pos_info in SPLIT_MARKER:
                sent += "、"
                f_words.append("、")
                words.append("、")
                pos.append("読点")
                pros.append("、")
        if boundary_label in STOP_MARKER:
            sent += "。"
            f_words.append("。")
            words.append("。")
            pos.append("句点")    
            pros.append("。")
            if "x" not in words:
                f_words.append("<EOS>")
                words.append("<EOS>")
                pos.append("<EOS>")
                pros.append("<EOS>")
                labels = make_label_seq(f_words)
                if len(labels)!=len(words):
                    print("f_words", len(f_words))
                    print("words", len(words))
                    print("labels", len(labels))
                assert len(words)==len(pos)==len(pros)==len(labels), xml_file 
                sents.append(sent)
                seq_words.append(words)
                seq_pos.append(pos)
                seq_pros.append(pros)
                seq_labels.append(labels)
            sent = ""
            f_words = ["<BOS>"]
            words = ["<BOS>"]
            pos = ["<BOS>"]
            pros = ["<BOS>"]    
        elif boundary_label is not None:
            sent += "、"
            f_words.append("、")
            words.append("、")
            pos.append("読点")
            pros.append("、")

    assert len(sents)==len(seq_words)==len(seq_pos)==len(seq_pros)==len(seq_labels)
    return sents, seq_words, seq_pos, seq_pros, seq_labels 


def main():
    all_sents = []
    all_words = []
    all_pos = []
    all_pros = []
    all_labels = []

    file_paths = get_files()
    for xml_file in file_paths:
        assert os.path.exists(xml_file)
        sents, words, pos, pros, labels = parse_xml(xml_file)
        all_sents += sents
        all_words += words
        all_pos += pos
        all_pros += pros
        all_labels += labels

    # save data
    out_path = Path(args.out_dir) / Path("sequences.txt")
    with open(out_path, mode="w") as f:
        for i in range(len(all_sents)):
            f.write("SENT {}\n".format(i))
            f.write("{}\n".format(all_sents[i]))
            f.write("w:{}\n".format(all_words[i]))
            f.write("p:{}\n".format(all_pos[i]))
            f.write("r:{}\n".format(all_pros[i]))
            f.write("L:{}\n".format(all_labels[i]))

    out_path = Path(args.out_dir) / Path("sequences.pkl")
    data = {
        "words": all_words,
        "pos": all_pos,
        "pronuns": all_pros,
        "labels": all_labels
    }
    with open(out_path, mode="wb") as f:
        pickle.dump(data, f) 


if __name__ == "__main__":
    args = get_args()
    main()
