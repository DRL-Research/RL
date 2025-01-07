import gym
import random
import numpy as np

def check_seed(expected_seed):
    # Check gym seed
    gym_seed, _ = gym.utils.seeding.np_random(expected_seed)
    gym_seed_correct = gym_seed == expected_seed

    # Check random seed
    random_seed_correct = random.getstate()[1][0] == expected_seed

    # Check numpy seed
    np_seed_correct = np.random.get_state()[1][0] == expected_seed

    return {
        "gym_seed_correct": gym_seed_correct,
        "random_seed_correct": random_seed_correct,
        "np_seed_correct": np_seed_correct
    }