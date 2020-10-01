"""
Add filler/error noise
"""
import os
import pickle
import argparse
from pathlib import Path
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="add_noise.py")
    parser.add_argument("--noise_model", required="True")
    parser.add_argument("--token_file", required="True")
    parser.add_argument("--label_file", required="True")
    parser.add_argument("--out_path", required="True")
    return parser.parse_args()


def add_noise(token, label, noise_dict):
    filler_list = noise_dict["<F>"][0]
    filler_prob = noise_dict["<F>"][1]
    error_list = noise_dict["<D>"][0]
    error_prob = noise_dict["<D>"][1]

    noise_token = []
    for i in range(len(label)):
        noise_token.append(token[i])
        if label[i] == "<F>":
            t =  np.random.choice(a=filler_list, p=filler_prob, size=1).tolist()[0]
            noise_token.append(t)
        elif label[i] == "<D>":
            t =  np.random.choice(a=filler_list, p=filler_prob, size=1).tolist()[0]
            noise_token.append(t)
    noise_token += token[len(label):]
    noise_token.remove("<BOS>")
    noise_token.remove("<EOS>")

    return noise_token


def main():
    assert os.path.exists(args.noise_model)
    assert os.path.exists(args.token_file)
    assert os.path.exists(args.label_file)

    noise_model = open(args.noise_model, mode="rb")
    token_file = open(args.token_file)
    label_file = open(args.label_file)

    noise_dict = pickle.load(noise_model)
    token_l = token_file.readlines()
    label_l = label_file.readlines()
    noise_token_l = []
    for token, label in zip(token_l, label_l):
        token = token.rstrip("\r\n").split()
        label = label.rstrip("\r\n").split()
        noise_token = add_noise(token, label, noise_dict)
        noise_token_l.append(noise_token)

    # save data
    out_path = Path(args.out_path+".noise")
    with open(out_path, mode="w") as f:
        for n in noise_token_l:
            f.write(" ".join(n)+"\n")


if __name__ == "__main__":
    args = get_args()
    main()
