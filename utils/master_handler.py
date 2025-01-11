import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd


class MasterNetwork(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(MasterNetwork, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size

        # רשת פשוטה יותר עם 3 שכבות
        self.layer1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.layer3 = nn.Linear(32, embedding_size)

        self.dropout = nn.Dropout(0.1)  # דרופאאוט קטן יותר

    def forward(self, x):
        # טיפול בצורת הקלט
        if len(x.shape) == 3:
            x = x.squeeze(1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)

        # וידוא גודל הקלט
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input features to be {self.input_size}, got {x.shape[-1]}")

        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = torch.tanh(self.layer3(x))  # tanh לנרמול האמבדינג

        return x


class MasterModel:
    def __init__(self, input_size, embedding_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = MasterNetwork(input_size, embedding_size).to(self.device)

        # אדם רגיל במקום Adamax
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=0.001,
            weight_decay=1e-4  # L2 regularization
        )

        # שינוי הפרמטרים של הScheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # שינוי ל-min כי אנחנו רוצים להקטין את הלוס
            factor=0.5,
            patience=2,
            verbose=True
        )

        self.metrics = {"loss": [], "return": []}

    def compute_loss(self, embeddings, returns):
        """
        לוס פשוט יותר שמנסה לקרב אמבדינגים של מצבים עם תגמולים דומים
        """
        # נרמול האמבדינגים
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # נרמול התגמולים לטווח [0,1]
        returns = (returns - returns.min()) / (returns.max() - returns.min() + 1e-8)

        # מטריצת מרחקים בין האמבדינגים
        dist_matrix = torch.cdist(embeddings, embeddings)

        # מטריצת הבדלי תגמולים
        rewards_dist = torch.abs(returns.unsqueeze(0) - returns.unsqueeze(1))

        # הלוס מנסה להתאים בין המרחקים
        loss = F.mse_loss(dist_matrix, rewards_dist)

        return loss

    def train_master(self, episode_states, episode_rewards):
        self.network.train()

        episode_embeddings = []
        for states in episode_states:
            # Debug information
            # print("Type of states:", type(states))
            # print("First state shape:", np.array(states[0]).shape if states else "Empty states")

            # Convert list of states to tensor properly
            try:
                states_batch = np.array([np.array(s).flatten() for s in states])
                states_tensor = torch.tensor(states_batch, dtype=torch.float32).to(self.device)

                # Get embeddings
                embeddings = self.network(states_tensor)
                avg_embedding = embeddings.mean(dim=0)
                episode_embeddings.append(avg_embedding)

            except Exception as e:
                print("Error processing states:")
                print("States type:", type(states))
                print("States content:", states)
                raise e

        episode_embeddings = torch.stack(episode_embeddings)
        rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float32).to(self.device)

        # חישוב הלוס ועדכון המשקולות
        self.optimizer.zero_grad()
        loss = self.compute_loss(episode_embeddings, rewards_tensor)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.scheduler.step(loss)

        self.metrics["loss"].append(loss.item())
        self.metrics["return"].append(sum(episode_rewards))
        print('master loss',loss.item())
        return loss.item()

    def get_proto_action(self, state):
        self.network.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            proto_action = self.network(state)
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