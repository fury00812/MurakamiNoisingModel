import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from build_vocab import PAD
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiRNN(nn.Module):
    def __init__(self, words_vocab, words_dim, pos_vocab, pos_dim,
            pros_vocab, pros_dim, hidden_dim, num_layers, dropout, labels_vocab):
        super(BiRNN, self).__init__()
        self.words_emb = nn.Embedding(words_vocab, words_dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(pos_vocab, pos_dim, padding_idx=PAD)
        self.pros_emb = nn.Embedding(pros_vocab, pros_dim, padding_idx=PAD)
        self.input_dim = words_dim + pos_dim + pros_dim
        self.rnn = nn.RNN(self.input_dim, hidden_dim, num_layers, \
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim*2, labels_vocab)  # 2 for bidirection


    def forward(self, words, pos, pros, seq_lengths, hidden=None):
        words_emb = self.words_emb(words)
        pos_emb = self.pos_emb(pos)
        pros_emb = self.pros_emb(pros)
        input_emb = torch.cat((words_emb, pos_emb, pros_emb), dim=2)
        packed = pack_padded_sequence(input_emb, lengths=seq_lengths,\
                            batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.linear(output)  # [batch_size * seq_len * label_values]
        return output
