import re
import collections
import random
import torch
from torch import nn
from torch.nn import functional as F

with open('../data/words.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [re.sub('[^\w\u4e00-\u9fff]+', ' ', line).strip() for line in lines]

tokens = [list(line) for line in lines]

class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq]), key=lambda x:' '.join(map(str, x))))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
    
vocab = Vocab(tokens)
corpus = [vocab[token] for line in tokens for token in line]

batch_size, num_steps = 64, 16

class SeqDataLoader:
    def __init__(self, corpus, batch_size, num_steps):
        self.corpus = corpus
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.data_iter_fn = self.seq_data_iter_sequential

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    
    def seq_data_iter_sequential(self, corpus, batch_size, num_steps):
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset: offset + num_tokens])
        Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps
        for i in range(0, num_steps * num_batches, num_steps):
            X = Xs[:, i: i + num_steps]
            Y = Ys[:, i: i + num_steps]
            yield X, Y

train_iter = SeqDataLoader(corpus, batch_size, num_steps)
vocab_size, num_hiddens, num_layers = len(vocab), 128, 2
device = torch.device('cuda')
lstm_layer = nn.LSTM(vocab_size, num_hiddens, num_layers)

class Model(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_layers, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(self.vocab_size, self.num_hiddens, self.num_layers)
        self.num_directions = 1
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.lstm(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state
    
    def begin_state(self, device, batch_size = 1):
        return (torch.zeros((self.num_directions * self.lstm.num_layers, batch_size, self.num_hiddens), device=device),
                torch.zeros((self.num_directions * self.lstm.num_layers, batch_size, self.num_hiddens), device=device))
    
net = Model(vocab_size, num_hiddens, num_layers, device)
net = net.to(device)

def predict(words, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[words[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in words[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# print(predict('新年快乐', 30, net, vocab, device))

num_epochs, lr, theta = 500, 2, 1e0
trainer = torch.optim.SGD(net.parameters(), lr)
loss = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    state = None
    net.train()
    for X, Y in train_iter:
        if state is None:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            for s in state:
                s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        trainer.zero_grad()
        l = loss(y_hat, y.long())
        l.backward()
        params = [p for p in net.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm
        trainer.step()
    
    net.eval()
    with torch.no_grad():
        print(predict('新年快乐', 120, net, vocab, device))
