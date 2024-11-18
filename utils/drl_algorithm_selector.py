from stable_baselines3 import PPO, DQN, A2C

class ModelSelector:
    def __init__(self, model_type, env, config):
        self.model_type = model_type
        self.env = env
        self.config = config
        self.model = self.init_model()

    def init_model(self):
        # Filter config based on the model type
        if self.model_type == 'PPO':
            ppo_config = {key: self.config[key] for key in ['learning_rate', 'n_steps', 'batch_size', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm'] if key in self.config}
            return PPO(policy='MlpPolicy', env=self.env, verbose=1, **ppo_config)
        elif self.model_type == 'DQN':
            dqn_config = {key: self.config[key] for key in ['learning_rate', 'buffer_size', 'batch_size', 'gamma', 'exploration_fraction', 'exploration_final_eps', 'exploration_initial_eps', 'target_update_interval'] if key in self.config}
            return DQN(policy='MlpPolicy', env=self.env, verbose=1, **dqn_config)
        elif self.model_type == 'A2C':
            a2c_config = {key: self.config[key] for key in ['learning_rate', 'n_steps', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm'] if key in self.config}
            return A2C(policy='MlpPolicy', env=self.env, verbose=1, **a2c_config)
        else:
            raise ValueError("Unsupported model type")

    def get_model(self):
        return self.model
