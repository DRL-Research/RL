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

    def train_step(self, inputs, target_embeddings):
        self.network.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.network(inputs)

        # Calculate loss
        loss = self.criterion(outputs, target_embeddings)

        # Backward pass
        loss.backward()

        # Print loss
        print(f"Loss: {loss.item()}")

        # Update weights
        self.optimizer.step()

        # Debug weights
        print("Weights after update:")
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                print(f"{name}: mean={param.data.mean().item()}, std={param.data.std().item()}")

        return loss.item()

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
