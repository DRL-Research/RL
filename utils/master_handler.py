import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
#comit

class MasterNetwork(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(MasterNetwork, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size

        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, embedding_size)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return x


class MasterModel:
    def __init__(self, input_size, embedding_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = MasterNetwork(input_size, embedding_size).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=0.001
        )

        self.metrics = {"loss": [], "return": []}

    def compute_loss(self, embeddings, returns):

        normalized_returns = 2 * (returns - returns.min()) / (returns.max() - returns.min() + 1e-8) - 1


        sim_matrix = torch.mm(embeddings, embeddings.t())


        returns_sim = torch.mm(normalized_returns.unsqueeze(1), normalized_returns.unsqueeze(0))
        print(sim_matrix,returns)

        loss = F.mse_loss(sim_matrix, returns_sim)

        return loss

    def train_master(self, episode_states, episode_rewards):
        self.network.train()

        total_loss = 0
        episodes_processed = 0

        for i, states in enumerate(episode_states):
            if not states:
                continue
            states_tensor = torch.stack([s.to(self.device) for s in states])

            if len(states_tensor.shape) == 3:
                states_tensor = states_tensor.squeeze(1)

            embeddings = self.network(states_tensor)
            #print('emb',embeddings)
            reward = float(episode_rewards[i][0])
            batch_rewards = torch.full((embeddings.shape[0],), reward).to(self.device)

            loss = self.compute_loss(embeddings, batch_rewards)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            episodes_processed += 1

            #print(f"Episode {i} - Loss: {loss.item():.6f}, Reward: {reward:.6f}")

        avg_loss = total_loss / max(episodes_processed, 1)
        self.metrics["loss"].append(avg_loss)
        self.metrics["return"].append(float(sum(reward[0] for reward in episode_rewards)))

        print(f"Master Loss: {avg_loss:.6f}, Return: {sum(reward[0] for reward in episode_rewards):.6f}")
        return avg_loss

    def get_proto_action(self, state):
        self.network.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            proto_action = self.network(state)
            print('proto aaction', proto_action)
            return proto_action.cpu().numpy().flatten()

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        self.network.eval()

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True
        self.network.train()

    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']