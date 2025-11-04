"""
QMIX Agent
Handles action selection, training, and epsilon-greedy exploration
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.qmix.qmix_networks import QMIXNetwork
from src.qmix.qmix_buffer import ReplayBuffer, EpisodeData, DemonstrationBuffer
import copy


class QMIXAgent:
    """
    QMIX Agent with centralized training and decentralized execution.
    """
    
    def __init__(
        self,
        obs_dim,
        state_dim,
        n_agents,
        n_actions,
        lr=5e-4,
        gamma=0.99,
        epsilon=0.1,  # Fixed epsilon as in paper
        buffer_size=5000,
        batch_size=32,
        target_update_interval=200,
        hidden_dim=64,
        mixing_embed_dim=32,
        device='cpu'
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = device
        
        # Networks
        self.qmix_net = QMIXNetwork(
            obs_dim, state_dim, n_agents, n_actions,
            hidden_dim, mixing_embed_dim
        ).to(device)
        
        self.target_qmix_net = copy.deepcopy(self.qmix_net)
        self.target_qmix_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.qmix_net.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.update_count = 0
        self.episode_count = 0
        
    def select_actions(self, observations, epsilon=None):
        """
        Select actions for all agents using epsilon-greedy.
        
        Args:
            observations: [n_agents, obs_dim] numpy array
            epsilon: exploration rate (uses self.epsilon if None)
        Returns:
            actions: [n_agents] numpy array of discrete actions
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            # Random actions
            actions = np.random.randint(0, self.n_actions, size=self.n_agents)
        else:
            # Greedy actions
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).to(self.device)
                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dim
                
                q_values = self.qmix_net.forward(obs_tensor, state=None)
                actions = q_values.argmax(dim=2).squeeze(0).cpu().numpy()
                
        return actions
        
    def train(self):
        """
        Perform one training step.
        Returns loss value or None if not enough data.
        """
        if not self.replay_buffer.can_sample(self.batch_size):
            return None
            
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        filled = torch.FloatTensor(batch['filled']).to(self.device)
        
        batch_size, max_len = rewards.shape
        
        # Compute current Q-values
        q_values = []
        q_totals = []
        
        for t in range(max_len):
            obs_t = obs[:, t]  # [batch, n_agents, obs_dim]
            state_t = states[:, t]  # [batch, state_dim]
            actions_t = actions[:, t]  # [batch, n_agents]
            
            q_t = self.qmix_net.forward(obs_t, state_t)  # [batch, n_agents, n_actions]
            q_total_t = self.qmix_net.get_q_total(q_t, state_t, actions_t)  # [batch, 1]
            
            q_values.append(q_t)
            q_totals.append(q_total_t)
            
        q_totals = torch.stack(q_totals, dim=1).squeeze(-1)  # [batch, max_len]
        
        # Compute target Q-values
        with torch.no_grad():
            q_targets = []
            
            for t in range(max_len):
                obs_t = obs[:, t]
                state_t = states[:, t]
                
                # Get target Q-values
                target_q_t = self.target_qmix_net.forward(obs_t, state_t)
                # Max over actions for each agent
                target_actions_t = target_q_t.argmax(dim=2)
                target_q_total_t = self.target_qmix_net.get_q_total(
                    target_q_t, state_t, target_actions_t
                )
                
                q_targets.append(target_q_total_t)
                
            q_targets = torch.stack(q_targets, dim=1).squeeze(-1)  # [batch, max_len]
            
            # Compute TD targets
            targets = rewards + self.gamma * q_targets * (1 - dones)
            
        # Compute TD loss (only for valid timesteps)
        td_error = (q_totals - targets) * filled
        loss = (td_error ** 2).sum() / filled.sum()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qmix_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
            
        return loss.item()
        
    def save(self, path):
        """Save model weights"""
        torch.save({
            'qmix_net': self.qmix_net.state_dict(),
            'target_qmix_net': self.target_qmix_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'episode_count': self.episode_count
        }, path)
        
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.qmix_net.load_state_dict(checkpoint['qmix_net'])
        self.target_qmix_net.load_state_dict(checkpoint['target_qmix_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint.get('update_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)


class QMIXwDAgent(QMIXAgent):
    """
    QMIX with Demonstrations (QMIXwD)
    Adds pre-training stage with expert and interaction data.
    """
    
    def __init__(
        self,
        obs_dim,
        state_dim,
        n_agents,
        n_actions,
        expert_ratio=0.1,  # 10% expert, 90% interaction
        lambda_demo=1.0,   # Weight for demonstration loss
        lambda_td=1.0,     # Weight for TD loss
        lambda_l2=1e-5,    # Weight for L2 regularization
        margin=0.8,        # Margin for supervised loss
        **kwargs
    ):
        super().__init__(obs_dim, state_dim, n_agents, n_actions, **kwargs)
        
        self.expert_ratio = expert_ratio
        self.lambda_demo = lambda_demo
        self.lambda_td = lambda_td
        self.lambda_l2 = lambda_l2
        self.margin = margin
        
        # Demonstration buffer
        self.demo_buffer = DemonstrationBuffer(
            buffer_size=kwargs.get('buffer_size', 5000),
            expert_ratio=expert_ratio
        )
        
        self.is_pretraining = False
        
    def supervised_margin_loss(self, q_values, actions_demo):
        """
        Compute supervised margin loss.
        Ensures Q(demo_action) >= max_a Q(a) - margin
        
        Args:
            q_values: [batch, n_agents, n_actions]
            actions_demo: [batch, n_agents] demonstrated actions
        Returns:
            loss: scalar
        """
        batch_size = q_values.shape[0]
        
        # Get Q-values for demonstrated actions
        actions_demo = actions_demo.long()
        q_demo = torch.gather(q_values, dim=2, index=actions_demo.unsqueeze(2)).squeeze(2)
        # [batch, n_agents]
        
        # Get max Q-values
        q_max, _ = q_values.max(dim=2)  # [batch, n_agents]
        
        # Margin loss: max(0, q_max - q_demo - margin)^2
        loss = F.relu(q_max - q_demo - self.margin) ** 2
        loss = loss.mean()
        
        return loss
        
    def train_with_demonstrations(self):
        """
        Training step during pre-training phase.
        Uses demonstration data with supervised margin loss, TD(λ), and L2 regularization.
        """
        if len(self.demo_buffer) < self.batch_size:
            return None
            
        # Sample from demonstration buffer
        batch = self.demo_buffer.sample(self.batch_size)
        if batch is None:
            return None
            
        # Convert to tensors
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        filled = torch.FloatTensor(batch['filled']).to(self.device)
        
        batch_size, max_len = rewards.shape
        
        # Compute Q-values and TD loss
        q_totals = []
        demo_losses = []
        
        for t in range(max_len):
            obs_t = obs[:, t]
            state_t = states[:, t]
            actions_t = actions[:, t]
            
            q_t = self.qmix_net.forward(obs_t, state_t)
            q_total_t = self.qmix_net.get_q_total(q_t, state_t, actions_t)
            
            q_totals.append(q_total_t)
            
            # Supervised margin loss (only on individual Q-values)
            demo_loss_t = self.supervised_margin_loss(q_t, actions_t)
            demo_losses.append(demo_loss_t)
            
        q_totals = torch.stack(q_totals, dim=1).squeeze(-1)
        demo_loss = torch.stack(demo_losses).mean()
        
        # TD(λ) loss
        with torch.no_grad():
            q_targets = []
            for t in range(max_len):
                obs_t = obs[:, t]
                state_t = states[:, t]
                target_q_t = self.target_qmix_net.forward(obs_t, state_t)
                target_actions_t = target_q_t.argmax(dim=2)
                target_q_total_t = self.target_qmix_net.get_q_total(
                    target_q_t, state_t, target_actions_t
                )
                q_targets.append(target_q_total_t)
            q_targets = torch.stack(q_targets, dim=1).squeeze(-1)
            targets = rewards + self.gamma * q_targets * (1 - dones)
            
        td_error = (q_totals - targets) * filled
        td_loss = (td_error ** 2).sum() / filled.sum()
        
        # L2 regularization
        l2_loss = sum(p.pow(2).sum() for p in self.qmix_net.parameters())
        
        # Total loss
        total_loss = (
            self.lambda_demo * demo_loss +
            self.lambda_td * td_loss +
            self.lambda_l2 * l2_loss
        )
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qmix_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
            
        return {
            'total_loss': total_loss.item(),
            'demo_loss': demo_loss.item(),
            'td_loss': td_loss.item(),
            'l2_loss': l2_loss.item()
        }
        
    def pretrain(self, n_steps):
        """
        Pre-training phase using demonstration data.
        
        Args:
            n_steps: Number of gradient updates
        Returns:
            losses: List of loss dictionaries
        """
        self.is_pretraining = True
        losses = []
        
        print(f"Starting pre-training for {n_steps} steps...")
        for step in range(n_steps):
            loss_dict = self.train_with_demonstrations()
            if loss_dict is not None:
                losses.append(loss_dict)
                
            if (step + 1) % 100 == 0:
                print(f"Pre-training step {step+1}/{n_steps}, Loss: {loss_dict}")
                
        self.is_pretraining = False
        print("Pre-training completed!")
        
        return losses
