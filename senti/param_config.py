import torch
import torch.nn as nn


def configure_senti_LSTM(vocab_to_int):
    nn_params = {}
    nn_params['vocab_size'] = len(vocab_to_int)+1
    nn_params['output_size'] = 1
    nn_params['embedding_dim'] = 400
    nn_params['hidden_dim'] = 256
    nn_params['n_layers'] = 2
    return nn_params


def configure_training_params(net):
    training_params = {}
    training_params['lr'] = 0.001
    training_params['criterion'] = nn.BCELoss()
    training_params['optimizer'] = torch.optim.Adam(net.parameters(), lr=training_params['lr'])
    training_params['epochs'] = 4
    training_params['print_every'] = 100
    training_params['clip'] = 5
    return training_params


def configure_utility_params():
    util_params = {}
    util_params['seq_length'] = 200
    util_params['split_frac'] = 0.8
    util_params['batch_size'] = 50
    return util_params
