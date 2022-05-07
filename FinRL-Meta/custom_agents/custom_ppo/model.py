from unicodedata import bidirectional
from torch import nn
from torch.distributions import normal
import torch

class Actor(nn.Module):

    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.mu = nn.Linear(in_features=64, out_features=self.n_actions)

        self.log_std = nn.Parameter(torch.zeros(1, self.n_actions))

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)

        std = self.log_std.exp()
        dist = normal.Normal(mu, std)

        return dist


class Critic(nn.Module):
    def __init__(self, n_states):
        super(Critic, self).__init__()
        self.n_states = n_states

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.value = nn.Linear(in_features=64, out_features=1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.value(x)

        return value


class ActorLSTM(nn.Module):

    def __init__(self, n_states, n_actions):
        super(ActorLSTM, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.lstm = nn.LSTM(input_size=4, hidden_size=28, num_layers=2, proj_size = 4,
                                     batch_first=True, bidirectional=True)
                                     #4 number of tech indicators

        self.fc1 = nn.Linear(in_features=10, out_features=64) # 4*2 indicators (*2 because of 2 layers) and 2 other features
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.mu = nn.Linear(in_features=64, out_features=self.n_actions)

        self.log_std = nn.Parameter(torch.zeros(1, self.n_actions))

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, inputs):
        x1, x2 = self.split_features(inputs)
        out, (hn, cn) = self.lstm(x2)
        out = out[:, -1, :]
        out = out.view(out.shape[0], -1)
        x = torch.cat((x1, out), dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        std = self.log_std.exp()
        dist = normal.Normal(mu, std) #approximation of the std deviation and mean of the normal
        #of the normal distribution and sample from here
        return dist

    def split_features(self, inputs):
        # split the state which is [cash, stock, tech_idx] to [cash, stock] and [tech_idx] with seq length
        x1 = inputs[:, :2] #cash stock
        x2 = inputs[:, 2:]
        x2 = x2.reshape(x2.shape[0], -1, 4) # 4 indicators
        return x1, x2


class CriticLSTM(nn.Module):

    def __init__(self, n_states):
        super(CriticLSTM, self).__init__()
        self.n_states = n_states
        
        self.lstm = nn.LSTM(input_size=4, hidden_size=28, num_layers=2, proj_size = 4,
                                     batch_first=True, bidirectional=True)
                                     #4 number of tech indicators
        self.fc1 = nn.Linear(in_features=10, out_features=64) # 4*2 indicators (*2 because of 2 layers) and 2 other features
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.value = nn.Linear(in_features=64, out_features=1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, inputs):
        x1, x2 = self.split_features(inputs)
        out, (hn, cn) = self.lstm(x2)
        out = out[:, -1, :]
        out = out.view(out.shape[0], -1)
        x = torch.cat((x1, out), dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.value(x)
        return x

    def split_features(self, inputs):
        # split the state which is [cash, stock, tech_idx] to [cash, stock] and [tech_idx] with seq length
        x1 = inputs[:, :2] #cash stock
        x2 = inputs[:, 2:]
        x2 = x2.reshape(x2.shape[0], -1, 4) # 4 indicators
        return x1, x2
