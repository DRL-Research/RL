from stable_baselines3 import PPO, DQN, A2C
from model.model_constants import ModelType, Policy


class Model:
    def __init__(self, env, experiment_config):
        self.env = env
        self.experiment_config = experiment_config
        self.model = self.init_model()

    def init_model(self):
        model_params = self.define_model_params(self.experiment_config)
        match self.experiment_config.MODEL_TYPE:
            case ModelType.PPO:
                print(PPO(policy=Policy, env=self.env, verbose=1, **model_params).learning_rate)
                return PPO(policy=Policy, env=self.env, verbose=1, **model_params)
            case ModelType.DQN:
                return DQN(policy=Policy, env=self.env, verbose=1, **model_params)
            case ModelType.A2C:
                return A2C(policy=Policy, env=self.env, verbose=1, **model_params)
            case _:
                raise ValueError(f"{self.experiment_config.MODEL_TYPE} Unsupported model type")

    @staticmethod
    def define_model_params(experiment):
        model_params = {
            'learning_rate': experiment.LEARNING_RATE,
            'n_steps': experiment.N_STEPS,
            'batch_size': experiment.BATCH_SIZE
        }
        return model_params
