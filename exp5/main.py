from src.constants import Role, CarName, ModelType
from src.experiment_config import Experiment
from training_loop import run_experiment

if __name__ == "__main__":
    # Create an Experiment instance. Do not pass computed fields.
    experiment_config = Experiment(
        EXPERIMENT_ID="Experiment5",
        ONLY_INFERENCE=False,
        EPISODE_AMOUNT_FOR_TRAIN=4,
        MODEL_TYPE=ModelType.PPO,
        ROLE=Role.BOTH,
        EPOCHS=10,
        LEARNING_RATE=0.075,
        N_STEPS=22,
        BATCH_SIZE=32,
        TIME_BETWEEN_STEPS=0.75,
        LOSS_FUNCTION="mse",
        EXPLORATION_EXPLOTATION_THRESHOLD=100,

    )
    run_experiment(experiment_config)
