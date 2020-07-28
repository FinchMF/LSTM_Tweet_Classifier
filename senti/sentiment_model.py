from string import punctuation

import torch
import torch.nn as nn


class Senti_LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim,
                    n_layers, drop_prob=0.5):
        super(Senti_LSTM, self).__init__()

        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc=nn.Linear(hidden_dim, output_size)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x, hidden):

        batch_size=x.size(0)
        embeds=self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out=lstm_out.contiguous().view(-1, self.hidden_dim)
        out=self.dropout(lstm_out)
        out=self.fc(out)
        sig_out=self.sigmoid(out)
        sig_out=sig_out.view(batch_size, -1)
        sig_out=sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):

        weight=next(self.parameters()).data

        train_on_gpu=torch.cuda.is_available()

        if train_on_gpu:
            hidden=(weight.new(self.n_layers, batch_size,
            self.hidden_dim).zero_().cuda(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden=(weight.new(self.n_layers, batch_size,
            self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden    



def tokenize_input(test, vocab_to_int):
    test = test.lower()
    text = ''.join([char for char in test if char not in punctuation])
    words = text.split()
    tokens = []
    tokens.append([vocab_to_int[word] for word in words])
    return tokens


def predict(net, test, vocab_to_int, seq_length, pad_features):
    net.eval()
    tokens = tokenize_input(test, vocab_to_int)
    seq_length=seq_length
    features = pad_features(tokens, seq_length)
    feature_tensor = torch.from_numpy(features)
    batch_size = feature_tensor.size(0)
    h = net.init_hidden(batch_size)
    train_on_gpu=torch.cuda.is_available()
    if train_on_gpu:
        feature_tensor = feature_tensor.cuda()
    output, h = net(feature_tensor, h)
    pred = torch.round(output.squeeze())
    print('Prediction value, pre-rounding: {:.6}'.format(output.item()))

    if pred.item() == 1:
        print('Democratic Sentiment detected')
    else:
        print('Republican Sentiment detected')

    