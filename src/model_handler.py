import os

from stable_baselines3 import PPO, DQN, A2C

from src.constants import ModelType, Policy


class Model:
    def __init__(self, env, experiment_config):
        self.env = env
        self.experiment_config = experiment_config
        self.model = self.init_model()

    def init_model(self):
        model_params, policy_kwargs = self.define_model_params(self.experiment_config)
        match self.experiment_config.MODEL_TYPE:
            case ModelType.PPO:
                return PPO(policy=Policy.MlpPolicy, env=self.env,verbose=1,
            n_epochs=4,
            vf_coef=0.7,
            ent_coef=0.01,
            gae_lambda=0.95,
            max_grad_norm=0.75,
            clip_range=0.25,
            clip_range_vf=1 ,
            policy_kwargs=policy_kwargs, **model_params)
            case ModelType.DQN:
                return DQN(policy=Policy.MlpPolicy, env=self.env, verbose=1, **model_params)
            case ModelType.A2C:
                return A2C(policy=Policy.MlpPolicy, env=self.env, verbose=1, **model_params)
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
                        experiment.PPO_NETWORK_ARCHITECTURE
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

    @classmethod
    def load(cls, model_path, env, experiment_config):
        """
        Load a pre-trained model from the given model_path.
        This method uses the load() method of the underlying stable-baselines3 model.
        """
        match experiment_config.MODEL_TYPE:
            case ModelType.PPO:
                loaded_model = PPO.load(model_path, env=env)
            case ModelType.A2C:
                loaded_model = A2C.load(model_path, env=env)
            case ModelType.DQN:
                loaded_model = DQN.load(model_path, env=env)
            case _:
                raise ValueError("Unsupported model type for loading.")
        # Create a new instance and set the loaded model
        instance = cls(env, experiment_config)
        instance.model = loaded_model
        return instance


def get_latest_model(directory):
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


def get_model_path_from_experiment_name(experiment_name):
    full_directory = os.path.join("experiments", experiment_name, "trained_model")
    return full_directory
