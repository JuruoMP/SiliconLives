# encoding: utf-8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTM(nn.Module):

    def __init__(self, embedding_dim, vocab_size, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.hidden = self._init_hidden_state()

    def _init_hidden_state(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view((len(sentence), 1, -1)), self.hidden)
        output = self.linear(lstm_out.view(len(sentence), -1))
        log_prob = F.log_softmax(output)
        return log_prob

model = LSTM(EMBEDDING_DIM, len(word_to_ix), HIDDEN_DIM, len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
    total_loss = 0.0
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model._init_hidden_state()
        sentence = prepare_sequence(sentence, word_to_ix)
        tags = prepare_sequence(tags, tag_to_ix)
        log_probs = model(sentence)
        loss = loss_function(log_probs, tags)
        total_loss += loss
        loss.backward()
        optimizer.step()
    print(total_loss)
