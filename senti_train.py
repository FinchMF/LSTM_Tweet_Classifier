from collections import Counter
import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from senti.param_config import configure_utility_params, configure_senti_LSTM, configure_training_params 
from senti.text_utils import encode_words, encode_labels, parse_tweets, pad_features, read_in_txt_data
from senti.sentiment_model import Senti_LSTM, predict

data_root = './data/'
ext = '.txt'
name = 'mp_tweet'
tweets_txt = f'{data_root}{name}_text{ext}'
labels_txt = f'{data_root}{name}_label{ext}'

tweets, labels = read_in_txt_data(tweets_txt, labels_txt)
words, tweets_split = parse_tweets(tweets)
vocab_to_int, tweets_ints = encode_words(words, tweets_split)
encoded_labels = encode_labels(labels)

tweets_lens = Counter([len(x) for x in tweets_ints])
print(f"Zero-length Tweets: {tweets_lens[0]}")
print(f"Maximum Tweets length: {max(tweets_lens)}")
print('\n')
print(f'Number of Tweets before removing outliers: {len(tweets_ints)}')
non_zero_idx = [ii for ii, tweet in enumerate(tweets_ints) if len(tweet) != 0]
tweets_ints = [tweets_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
print('\n')
print(f'Number of Tweets after removing outliers: {len(tweets_ints)}')
print('\n')

utility_params = configure_utility_params()

features = pad_features(tweets_ints, utility_params['seq_length'])

assert len(features) == len(tweets_ints)
assert len(features[0]) == utility_params['seq_length']

split_idx = int(len(features)*utility_params['split_frac'])
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]
print(len(remaining_y))

test_idx = int(len(remaining_x)*0.5)
print(test_idx)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

print('\t\t\tFeature Shapes:')
print(f'Train set: \t\t{train_x.shape}',
      f'\nValidation set: \t{val_x.shape}',
      f'\nTest set: \t\t{test_x.shape}')
print('\n')
print('\t\t\tLabel Shapes:')
print(f'Train set: \t\t{train_y.shape}',
      f'\nValidation set: \t{val_y.shape}',
      f'\nTest set: \t\t{test_y.shape}')
print('\n')

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_data, shuffle=True, batch_size=utility_params['batch_size'], drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=utility_params['batch_size'], drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=utility_params['batch_size'], drop_last=True)

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print(f'Sample Input size: {sample_x.size()}')
print(f'Sample Input: \n {sample_x}')
print('\n')
print(f'Sample Label size: {sample_y.size()}')
print(f'Sample Label: \n {sample_y}')
print('\n')

train_on_gpu=torch.cuda.is_available()

if train_on_gpu:
    print('Training on GPU')
else: 
    print('No GPU available, training on CPU')

senti_lstm_params = configure_senti_LSTM(vocab_to_int)

net = Senti_LSTM(senti_lstm_params['vocab_size'],
                 senti_lstm_params['output_size'],
                 senti_lstm_params['embedding_dim'],
                 senti_lstm_params['hidden_dim'],
                 senti_lstm_params['n_layers']
                 )

print('Model Configuration: ')
print(net)

training_params = configure_training_params(net)

if train_on_gpu:
    net.cuda()

counter = 0
net.train()
for e in range(training_params['epochs']):
    h = net.init_hidden(utility_params['batch_size'])

    for inputs, labels in train_loader:
        counter += 1

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        h = tuple([each.data for each in h])

        net.zero_grad()

        output, h = net(inputs, h)
        loss = training_params['criterion'](output.squeeze(), labels.float())
        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), training_params['clip'])
        training_params['optimizer'].step()

        if counter % training_params['print_every'] == 0:
            val_h = net.init_hidden(utility_params['batch_size'])
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                 val_h = tuple([each.data for each in val_h])

                 if train_on_gpu:
                     inputs, labels = inputs.cuda(), labels.cuda()

                 output, val_h = net(inputs, val_h)
                 val_loss = training_params['criterion'](output.squeeze(), labels.float())

                 val_losses.append(val_loss.item())

            net.train()
            print(f'Epoch: {e+1}/{training_params["epochs"]}',
                  f'Step: {counter}',
                  'Loss: {:.6f}...'.format(loss.item()),
                  'Val Loss: {:.6}'.format(np.mean(val_losses)))

torch.save(net.state_dict, f'./models/{name}_senti_net_params.pth')
print('Senti_Net Classifier\'s parameters saved....')
torch.save(net, f'./models/{name}_senti_net_classifer.pth')
print('Senti_Net Classifier Network saved....')        


####################
# Testing Accuracy #
####################

republican_test = 'Trump is my president'
democratic_test = 'Bernie Sanders!!'


Poli_Senti_Net = torch.load(f'./models/mp_tweet_senti_net_classifer.pth')
print('Trained Model Loaded....')
print('\n')
print(f'Test 1: \
        republican_tweet: {republican_test}')
print('\n')
print('Prediction:')
predict(Poli_Senti_Net, republican_test, vocab_to_int, utility_params['seq_length'], pad_features)
print('\n')
print(f'Test 2: \
        democratic_tweet: {Democratic_test}')
print('\n')
print('Prediction:')
predict(Poli_Senti_Net, democratic_test, vocab_to_int, utility_params['seq_length'], pad_features)