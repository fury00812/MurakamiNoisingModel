"""
Train the sequence labeling model
"""
import os
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from dataset import DataSet
from dataset_gen import DataSetGen
from build_vocab import PAD
from build_vocab import Dictionary
from birnn import BiRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--generate", action="store_true",
                        help="Generation mode")
    if not parser.parse_known_args()[0].generate:
        parser.add_argument("--train_dataset", required="True")
        parser.add_argument("--valid_dataset", required="True")
        parser.add_argument("--out_dir", required="True")
    else:
        parser.add_argument("--out_path", required="True")
    parser.add_argument("--test_dataset")
    parser.add_argument("--dicts", required="True",
                        help="dictionaries of 4 types of Dictionary objects")
    parser.add_argument("--save_path", default="",
                        help="checkpoint name saved in out_dir")
    parser.add_argument("--max_vocab", type=int, default=16000,
                        help="maximum vocabulary size of words")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--valid_epoch", type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=-1,
                        help="Number of iterations to continue training\
                            without improvement in validation loss")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--words_dim", type=int, default=200,
                        help="embeddig dim of word sequences")
    parser.add_argument("--pos_dim", type=int, default=50,
                        help="embedding dim of POS sequences")
    parser.add_argument("--pros_dim", type=int, default=50,
                        help="embedding dim of pronunciation sequences")
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluations")
    return parser.parse_args()


def masked_cross_entropy(hyp, ref, seq_lengths):
    # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    mce = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD)
    # align ref size to hyp
    if len(hyp.view(-1, hyp.size(-1))) != len(ref.view(-1)):
        ref = pack_padded_sequence(ref, lengths=seq_lengths,\
                        batch_first=True, enforce_sorted=False)
        ref, _ = pad_packed_sequence(ref, batch_first=True)
    return mce(hyp.view(-1, hyp.size(-1)), ref.view(-1))


def compute_loss(words, pos, pros, seq_lengths, labels, model, optimizer, is_train=True):
    """
    Do inference, loss calculation, and update parameters if is_train=True
    """
    # inference
    model.train(is_train)
    hyp = model(words, pos, pros, seq_lengths)

    # calculate loss
    loss = masked_cross_entropy(hyp.contiguous(), labels.contiguous(), seq_lengths)

    # update parameters
    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def train(args, model):
    """
    train the Bidirectional RNN model
    """
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_valid_loss = 10000
    prev_valid_loss = 10000
    stop_counter = 0
    for epoch in range(1, args.num_epoch+1):
        train_loss = 0
        valid_loss = 0
        # train
        for batch in args.train_loader:
            words= batch["words"].to(device)  # tensor: [batch_size * seq_len]
            pos = batch["pos"].to(device)  # tensor: [batch_size * seq_len]
            pros = batch["pronuns"].to(device)  # tensor: [batch_size * seq_len]
            labels = batch["labels"].to(device)  # tensor: [batch_size * seq_len]
            seq_lengths = batch["len"].to(device)
            loss = compute_loss(words, pos, pros, seq_lengths, labels,\
                                        model, optimizer, is_train=True)
            train_loss += loss
        train_loss = train_loss / args.train_dataset.__len__()
        # valid
        if epoch % args.valid_epoch == 0:
            for batch in args.valid_loader:
                words= batch["words"].to(device)  # tensor: [batch_size * seq_len]
                pos = batch["pos"].to(device)  # tensor: [batch_size * seq_len]
                pros = batch["pronuns"].to(device)  # tensor: [batch_size * seq_len]
                labels = batch["labels"].to(device)  # tensor: [batch_size * seq_len]
                seq_lengths = batch["len"].to(device)
                loss = compute_loss(words, pos, pros, seq_lengths, labels,\
                                            model, optimizer, is_train=False)
                valid_loss += loss
            valid_loss = valid_loss / args.valid_dataset.__len__()
            print("Epoch {}: train_loss: {:5.2f} valid_loss: {:5.2f}"
                            .format(epoch, train_loss, valid_loss))
            if valid_loss < best_valid_loss:
                args.save_path = Path(args.out_dir) / Path("best_epoch.pth")
                print("Updated the best valid loss. saving to: {}"
                            .format(args.save_path))
                torch.save(model.state_dict(), args.save_path)
                best_valid_loss = valid_loss
                stop_counter = 0
            elif args.early_stopping > 0:
                stop_counter += 1
                print("Not a better valid loss ({}/{})."
                        .format(stop_counter, args.early_stopping))
                if stop_counter >= args.early_stopping:
                    print("Stopping criterion has been below its best value "\
                          "{} times in a row. Finish training.".format(stop_counter))
                    break

        else:
            print("Epoch {}: train_loss: {:5.2f}".format(epoch, train_loss))
    save_path = Path(args.out_dir) / Path("last_epoch.pth")
    print("Finished training. saving the last model to: {}"
                .format(save_path))
    torch.save(model.state_dict(), save_path)


def calc_accuracy(hyp_l, ref_l):
    total_correct = 0
    noise_correct = 0
    num_tokens = 0
    noise_tokens = 0
    for hyp_sent, ref_sent in zip(hyp_l, ref_l):
        for hyp, ref in zip(hyp_sent, ref_sent):
            total_correct = total_correct+1 if hyp==ref else total_correct
            num_tokens+=1
            if ref!=args.dicts["labels"].word2id["O"]:
                noise_correct = noise_correct+1 if hyp==ref else noise_correct
                noise_tokens+=1
    accuracy = total_correct / num_tokens
    noise_accuracy = noise_correct / noise_tokens
    return accuracy, noise_accuracy


def generate(args, model):
    """
    Load trained model and generate outputs
    """
    src_l = []
    hyp_l = []
    len_l = []
    model.load_state_dict(torch.load(args.save_path))
    model.to(device)
    model.eval()
    for batch in args.test_loader:
        words= batch["words"].to(device)
        pos = batch["pos"].to(device)
        pros = batch["pronuns"].to(device)
        seq_lengths = batch["len"].to(device)
        hyp = model(words, pos, pros, seq_lengths)
        hyp = hyp.max(dim=-1)[1].data.cpu().tolist()
        src = words.cpu().tolist()
        for b in range(len(hyp)):
            for i in range(len(hyp[b])):
                hyp[b][i] = args.dicts["labels"].id2word[hyp[b][i]]
        for b in range(len(src)):
            for i in range(len(src[b])):
                src[b][i] = args.dicts["words"].id2word[src[b][i]]
        seq_len = seq_lengths.tolist()
        hyp_l += hyp
        src_l += src
        len_l += seq_len
    for i in range(len(hyp_l)):
        hyp_l[i] = hyp_l[i][:len_l[i]]
        src_l[i] = src_l[i][:len_l[i]]
        assert len(hyp_l[i])==len(src_l[i]), (hyp_l[i], src_l[i])

    return src_l, hyp_l


def evaluate(args, model):
    """
    Load trained model and generate outputs, then evaluate model
    """
    hyp_l = []
    ref_l = []
    len_l = []
    model.load_state_dict(torch.load(args.save_path))
    model.to(device)
    model.eval()
    for batch in args.test_loader:
        words= batch["words"].to(device)
        pos = batch["pos"].to(device)
        pros = batch["pronuns"].to(device)
        labels = batch["labels"].to(device)
        seq_lengths = batch["len"].to(device)
        hyp = model(words, pos, pros, seq_lengths)
        hyp = hyp.max(dim=-1)[1].data.cpu().tolist()
        ref = labels.cpu().tolist()
        seq_len = seq_lengths.tolist()
        hyp_l += hyp
        ref_l += ref
        len_l += seq_len
    for i in range(len(ref_l)):
        hyp_l[i] = hyp_l[i][:len_l[i]]
        ref_l[i] = ref_l[i][:len_l[i]]
    accuracy, noise_accuracy = calc_accuracy(hyp_l, ref_l)
    print("Test set accuracy (excerpt \"O\"): {:5.2f} ({:5.2f})"
                    .format(accuracy, noise_accuracy))


def main():

    # load Dictionary
    assert os.path.exists(args.dicts)
    with open(args.dicts, mode="rb") as f:
        args.dicts = pickle.load(f)
    args.words_vocab = min(args.max_vocab, args.dicts["words"].num_words)
    args.pos_vocab = args.dicts["pos"].num_words
    args.pros_vocab = args.dicts["pronuns"].num_words
    args.labels_vocab = args.dicts["labels"].num_words

    # build the model
    model_args = {
        "words_vocab": args.words_vocab,
        "words_dim": args.words_dim,
        "pos_vocab": args.pos_vocab,
        "pos_dim": args.pos_dim,
        "pros_vocab": args.pros_vocab,
        "pros_dim": args.pros_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "labels_vocab": args.labels_vocab
    }
    model = BiRNN(**model_args).to(device)
    print(model)

    # load testset
    if args.test_dataset is not None:
        assert os.path.exists(args.test_dataset)
        args.test_dataset = torch.load(args.test_dataset) 
        args.test_loader = DataLoader(\
                args.test_dataset, batch_size=args.batch_size)
        print("[INFO] Test data has {} sents.".format(args.test_dataset.__len__()))

    # get sequence length
    args.max_len=len(args.test_dataset.__getitem__(0)["words"])

    # generation (generate.sh / Phase 2)
    if args.generate:
        # generate label
        src, label = generate(args, model)
        # save files
        out_path = Path(args.out_path+".src")
        with open(out_path, mode="w") as f:
            for s in src:
                f.write((" ".join(s)+"\n"))
        out_path = Path(args.out_path+".label")
        with open(out_path, mode="w") as f:
            for s in label:
                f.write((" ".join(s)+"\n"))
        exit()

    # eval_only
    if args.eval_only:
        args.save_path = Path(args.out_dir) / Path(args.save_path)
        assert args.test_dataset is not None
        assert args.save_path is not None
        evaluate(args, model)
        exit()

    # load train & valid set
    assert os.path.exists(args.train_dataset)
    assert os.path.exists(args.valid_dataset)
    args.train_dataset = torch.load(args.train_dataset)
    args.valid_dataset = torch.load(args.valid_dataset)
    args.train_loader = DataLoader(\
                args.train_dataset, batch_size=args.batch_size)
    args.valid_loader = DataLoader(\
                args.valid_dataset, batch_size=args.batch_size)
    print("[INFO] Validation data has {} sents.".format(args.valid_dataset.__len__()))
    print("[INFO] Training data has {} sents.".format(args.train_dataset.__len__()))

    # train & evaluate
    train(args, model)
    evaluate(args, model)


if __name__ == "__main__":
    args = get_args()
    main()
