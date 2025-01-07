import torch
import torch.nn as nn

class MasterNetwork(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(MasterNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, 64)
        self.hidden_layer = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, embedding_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

class MasterModel:
    def __init__(self, input_size, embedding_size, learning_rate):
        self.network = MasterNetwork(input_size, embedding_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_master(self, states, rewards):
        """
        Train the Master network using policy gradient updates.
        :param states: The states observed during the episode (list of numpy arrays or tensors).
        :param rewards: The rewards collected during the episode (list of scalars).
        """
        self.network.train()
        self.optimizer.zero_grad()

        # Convert states to tensors and stack into a single tensor
        states_tensor = torch.stack(
            [torch.tensor(state, dtype=torch.float32) for state in states])  # Shape: (num_states, state_dim)
        print(f"States tensor shape: {states_tensor.shape}")

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)  # Shape: (num_states,)

        # Calculate discounted cumulative rewards (G_t)
        discounted_rewards = self.discount_rewards(rewards_tensor)

        # Forward pass through the network
        outputs = self.network(states_tensor)  # Shape: (num_states, embedding_dim)

        # Calculate loss using policy gradient (negative log likelihood scaled by rewards)
        log_probs = torch.log_softmax(outputs, dim=-1)  # Log probabilities
        loss = -torch.sum(log_probs * discounted_rewards.unsqueeze(1))  # Policy gradient loss

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        print(f"Master Loss: {loss.item()}")
        return loss.item()

    def discount_rewards(self, rewards, gamma=0.99):
        """
        Compute the discounted cumulative rewards for a given list of rewards.
        :param rewards: A tensor of rewards collected during the episode.
        :param gamma: Discount factor (default: 0.99).
        :return: Discounted cumulative rewards.
        """
        discounted = []
        cumulative = 0
        for reward in reversed(rewards):
            cumulative = reward + gamma * cumulative
            discounted.insert(0, cumulative)
        return torch.tensor(discounted, dtype=torch.float32)

    def inference(self, inputs):
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(inputs)
        return outputs

    def get_proto_action(self, inputs):
        self.network.eval()
        with torch.no_grad():
            embedding = self.network(inputs)
        proto_action = embedding.squeeze(0)
        return proto_action

    def save(self, path):
        torch.save(self.network.state_dict(), path)


    def load(self, path):
        self.network.load_state_dict(torch.load(path))
        self.network.eval()

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True
