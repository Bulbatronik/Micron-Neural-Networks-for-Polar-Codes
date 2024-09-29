import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, depth, hidden_size, skip_connect):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        previous_size = input_dim

        for i in range(depth):
            layer = nn.Linear(previous_size, 1 if i == depth - 1 else hidden_size)
            previous_size = layer.out_features
            if i < depth - 1:
                layer = nn.Sequential(layer, nn.ReLU())
            self.layers.append(layer)
        self.skip_connect = skip_connect


    def forward(self, x):
        activations = [x]
        for i in range(len(self.layers)):
            activations.append(self.layers[i](activations[-1]))
            if i - 1 % self.skip_connect == 0 and i > self.skip_connect:
                activations[-1] += activations[-1 - self.skip_connect]
        return activations[-1].reshape(-1)
    

    def fit(self, loader, epochs, verbose=True):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            total_loss = 0
            for i, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                y_pred = self(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/(i+1)}')


    @torch.no_grad()
    def test(self, loader, verbose=True):
        self.eval()
        criterion = nn.MSELoss()
        results = {'y_true': [], 'y_hat': []}
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            y_pred = self(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            results['y_true'].append(y)
            results['y_hat'].append(y_pred)
        if verbose:
            print(f'Test Loss: {total_loss/(i+1)}')
        return results