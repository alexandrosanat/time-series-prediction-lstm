import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def generate_data():
    n_samples = 100
    length_of_sample = 1000
    width_of_wave = 20

    x = np.empty((n_samples, length_of_sample), np.float32)
    # Shift each sample randomly on the x-axis
    x[:] = np.array(range(length_of_sample)) + np.random.randint(-4 * width_of_wave,
                                                                 4 * width_of_wave,
                                                                 n_samples).reshape(n_samples, 1)
    y = np.sin(x / 1.0 / width_of_wave).astype(np.float32)
    return x, y


def plot_graph(x, y, title):
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(np.arange(x.shape[1]), y[0, :], 'r', linewidth=2.0)  # Plot first sample only
    plt.show()


class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=50):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        # LSTM1, LSTM2, Linear
        # Input is 1 as we go over each sample's values one by one
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        """If future=0 do training on known values, else use
        future value to predict the next "future" no of values"""
        outputs = list()
        n_samples = x.size(0)

        # Specify initial states
        hidden_state1 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        cell_state1 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        hidden_state2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        cell_state2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        # Split each tensor in chunks of 1
        # Original tensor is (N, 1000) we slice to (N, 1)
        for input_t in x.split(1, dim=1):
            hidden_state1, cell_state1 = self.lstm1(input_t, (hidden_state1, cell_state1))
            hidden_state2, cell_state2 = self.lstm2(hidden_state1, (hidden_state2, cell_state2))
            output = self.linear(hidden_state2)
            outputs.append(output)

        for i in range(future):
            hidden_state1, cell_state1 = self.lstm1(output, (hidden_state1, cell_state1))
            hidden_state2, cell_state2 = self.lstm2(hidden_state1, (hidden_state2, cell_state2))
            output = self.linear(hidden_state2)
            outputs.append(output)

        # Concatenate outputs
        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    x, y = generate_data()
    # plot_graph(x, y, "Sine wave")

    # y = (100, 1000)
    # Use all but the first three samples for training
    train_input = torch.from_numpy(y[3:, :-1])  # (97, 999)
    train_target = torch.from_numpy(y[3:, 1:])  # (97, 999)
    test_input = torch.from_numpy(y[:3, :-1])  # (3, 999)
    test_target = torch.from_numpy(y[:3, 1:])  # (3, 999)

    model = LSTMPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    n_steps = 10

    for i in range(n_steps):
        print("Step: ", i)


        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target)
            print("Loss: ", loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print("Test loss: ", loss.item())
            y = pred.detach().numpy()

        plt.figure(figsize=(12, 6))
        plt.title(f"Step: {i + 1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        n = train_input.shape[1]  # 999

        def draw(y_i, colour):
            plt.plot(np.arange(n), y_i[:n], colour, linewidth=2.0)  # Plot actual values
            plt.plot(np.arange(n, n+future), y_i[n:], colour + ":", linewidth=2.0)  # Plot predictions

        draw(y[0], 'r')
        draw(y[1], 'b')
        draw(y[2], 'g')

        plt.savefig(f"plots/predict {i}.jpg")
        plt.close()
