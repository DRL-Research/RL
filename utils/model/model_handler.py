import os

from stable_baselines3 import PPO, DQN, A2C

from utils.model.model_constants import ModelType, Policy


class Model:
    def __init__(self, env, experiment_config):
        self.env = env
        self.experiment_config = experiment_config
        self.model = self.init_model()

    def init_model(self):
        model_params, policy_kwargs = self.define_model_params(self.experiment_config)
        match self.experiment_config.MODEL_TYPE:
            case ModelType.PPO:
                print(model_params)
                print(policy_kwargs)
                m = PPO(policy=Policy, env=self.env, verbose=1, policy_kwargs=policy_kwargs, **model_params)
                print("Custom policy network:", m.policy.mlp_extractor.policy_net)
                print("Custom value network:", m.policy.mlp_extractor.value_net)
                return m
            case ModelType.DQN:
                return DQN(policy=Policy, env=self.env, verbose=1, **model_params)
            case ModelType.A2C:
                return A2C(policy=Policy, env=self.env, verbose=1, **model_params)
            case _:
                raise ValueError(f"{self.experiment_config.MODEL_TYPE} Unsupported model type")

    @staticmethod
    def define_model_params(experiment):

        policy_kwargs = None

        common_params = {
            'learning_rate': experiment.LEARNING_RATE,
            'batch_size': experiment.BATCH_SIZE
        }

        match experiment.MODEL_TYPE:
            case ModelType.PPO:
                policy_kwargs = {
                    'net_arch': [
                        {'pi': [32, 32], 'vf': [32, 32]}
                    ]
                }
                model_params = {
                    **common_params,
                    'n_steps': experiment.N_STEPS
                }

            case ModelType.A2C:
                model_params = {
                    **common_params,
                    'n_steps': experiment.N_STEPS
                }

            case ModelType.DQN:
                model_params = common_params

            case _:
                raise ValueError(f"Unsupported model type: {experiment}")

        return model_params, policy_kwargs

    @staticmethod
    def get_latest_model(self, directory):
        '''

        Get last model from all directory. file must end with .zip
        :param directory:
        :return: last model
        '''
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(".zip"):
                    files.append(os.path.join(root, filename))

        if not files:
            return None

        latest_file = max(files, key=os.path.getctime)
        print('Latest model:', latest_file)
        return latest_file

    def get_model_from_specific_directory(self, directory):
        '''
        return the last model from specific directory

        :param directory:
        :return: last model
        '''
        relevant_directory = os.chdir(directory)
        for file in os.listdir(relevant_directory):
            if file.endswith(".zip"):
                return file
