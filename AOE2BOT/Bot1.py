import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import gym

# -------------- Step 1: Load & Convert Replays --------------
def load_replay_data():
    """Loads all AoE2 replays (converted to JSON) from a folder."""
    replay_files = glob.glob("replays/*.json")  # Assuming replays are stored as JSON
    all_games = []
    for replay in replay_files:
        with open(replay, "r") as f:
            game_data = json.load(f)
            all_games.append(game_data)
    return all_games

def extract_time_series_features(game_data):
    """Extracts time-series features from all frames in a replay."""
    resources = np.array(game_data["resources"])  # Shape: (T, 4) -> T frames, 4 resources
    army = np.array(game_data["army_composition"])  # Shape: (T, 3) -> T frames, 3 unit types
    actions = np.array(game_data["actions"])  # Shape: (T,)

    # Flatten time-series data while preserving temporal patterns
    return np.concatenate([resources.flatten(), army.flatten(), actions.flatten()])

# Load and extract features from all replays
game_data_list = load_replay_data()
feature_vectors = np.array([extract_time_series_features(game) for game in game_data_list])

# -------------- Step 2: Clustering Using DTW --------------
num_clusters = 5  # Number of strategy clusters
kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", random_state=42)
labels = kmeans.fit_predict(feature_vectors.reshape(len(feature_vectors), -1, 1))  # Reshape for DTW

# Visualize clusters
plt.hist(labels, bins=num_clusters)
plt.title("Strategy Clusters (DTW-based)")
plt.xlabel("Cluster")
plt.ylabel("Number of Games")
plt.show()

def assign_cluster_to_replay(game_data):
    """Assigns a replay to a strategy cluster using DTW clustering."""
    features = extract_time_series_features(game_data).reshape(1, -1, 1)
    return kmeans.predict(features)[0]

# -------------- Step 3: Define AoE2 Environment for RL --------------
class AoE2Env(gym.Env):
    """Custom RL environment for AoE2."""
    def __init__(self):
        super(AoE2Env, self).__init__()
        self.action_space = gym.spaces.Discrete(6)  # Train infantry, cavalry, archers, gather resources
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(11,), dtype=np.float32)  
        self.current_state = np.zeros(11, dtype=np.float32)

    def step(self, action):
        """Take an action and update the environment."""
        self.send_action_to_game(action)  # Send action to AoE2
        self.current_state = self.get_game_state()  # Get new game state
        reward = self.calculate_reward()  # Compute reward
        done = self.check_game_end()  # Check if game is over
        return self.current_state, reward, done, {}

    def reset(self):
        """Resets the game state at the start of training."""
        self.current_state = self.get_game_state()
        return self.current_state

    def get_game_state(self):
        """Reads game state from AoE2 files."""
        resources = self.read_resources_from_game()
        my_army = self.read_army_from_game()
        enemy_army = self.read_enemy_army_from_game()
        exploration = self.read_exploration_from_game()
        return np.array(resources + [exploration] + my_army + enemy_army, dtype=np.float32)

    def calculate_reward(self):
        """Calculates reward based on army strength and economy."""
        my_army_strength = sum(self.read_army_from_game())
        enemy_army_strength = sum(self.read_enemy_army_from_game())
        resource_score = sum(self.read_resources_from_game())
        return my_army_strength - enemy_army_strength + (resource_score / 100)

    def check_game_end(self):
        """Checks if the game is over based on army strength."""
        return sum(self.read_enemy_army_from_game()) == 0

    def send_action_to_game(self, action):
        """Writes the RL action to the game for execution."""
        actions = ["train_infantry", "train_cavalry", "train_archers", "gather_food", "gather_wood", "gather_gold"]
        with open("rl_commands.txt", "w") as f:
            f.write(actions[action])

    def read_resources_from_game(self):
        """Reads resource data from AoE2 state file."""
        try:
            with open("aoe2_state.txt", "r") as f:
                data = f.read().split(":")[1]
                return [int(x.strip()) for x in data.split(",")]
        except:
            return [0, 0, 0, 0]

    def read_army_from_game(self):
        """Reads player's army composition from AoE2."""
        try:
            with open("aoe2_army.txt", "r") as f:
                data = f.read().split("|")
                return [int(x.split(":")[1]) for x in data[0].split(",")]
        except:
            return [0, 0, 0]

    def read_enemy_army_from_game(self):
        """Reads enemy army composition."""
        try:
            with open("aoe2_army.txt", "r") as f:
                data = f.read().split("|")
                return [int(x.split(":")[1]) for x in data[1].split(",")]
        except:
            return [0, 0, 0]

    def read_exploration_from_game(self):
        """Reads fog of war exploration percentage."""
        try:
            with open("aoe2_state.txt", "r") as f:
                return int(f.read().split(":")[1].strip())
        except:
            return 0

# -------------- Step 4: Train PPO RL Agent --------------
env = DummyVecEnv([lambda: AoE2Env()])  # Wrap in DummyVecEnv
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# -------------- Step 5: Real-Time RL Execution in AoE2 --------------
import time

def run_rl_in_aoe2():
    """Runs the trained RL agent in a live AoE2 game."""
    env = AoE2Env()
    while True:
        state = env.get_game_state()
        action, _ = model.predict(state)
        env.send_action_to_game(action)
        time.sleep(1)  # Sync with game pace

run_rl_in_aoe2()
