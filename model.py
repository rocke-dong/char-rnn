import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = self.embedding(input)
        _, (h_n, c_n) = self.lstm(input)
        hidden = torch.mean(h_n, dim=0)
        output = self.out(hidden)
        return output


if __name__ == "__main__":
    from icecream import ic    
    # n_hidden = 128
    # rnn = RNN(n_letters, n_hidden, n_categories)
    # input = lineToTensor('Albert')
    # ic(input.shape)
    # output, next_hidden = rnn(input)
    # ic(output.shape)
    # ic(output)
    m = RNN(20, 10)
    input = torch.randint(19, (5, 20))
    ic(input.shape)
    a = m(input)
    ic(a)