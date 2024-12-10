import gymnasium as gym
from stable_baselines3 import PPO
from matplotlib import pyplot as plt

# קונפיגורציה מותאמת אישית לסביבה

# יצירת סביבה
env = gym.make("intersection-v0", render_mode="human", config=config)

# הגדרת המודל
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0005,
    n_steps=2048,
    batch_size=64,
    verbose=1,
)

# פונקציה להתאמת פעולות רכב 1 (הסוכן)
def set_agent_actions(env):
    env.config["action"]["type"] = "DiscreteMetaAction"
    env.reset()
    env.road.vehicles[0].target_speed = 0.5  # הגבלת מהירות רכב 1

# אימון המודל
def train_model():
    for episode in range(10):  # מספר פרקים
        obs, _ = env.reset()
        done = truncated = False
        episode_reward = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            episode_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    # שמירת המודל
    model.save("ppo_intersection_model")
    print("Model saved!")

# הרצת אימון
set_agent_actions(env)
train_model()
