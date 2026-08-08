import json
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def parse_army_state_vectors_and_fog(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Define class_id to unit group mapping
    unit_classes = {
        "infantry": [1],
        "cavalry": [3],
        "archers": [4],
        "siege": [15],
        "monks": [10],
    }

    results = {}

    for player in data['players']:
        name = player['name']
        army_counts = defaultdict(int)
        explored_tiles = set()

        for obj in player.get("objects", []):
            class_id = obj.get("class_id", None)
            pos = obj.get("position", {})

            # Group by class
            for group, ids in unit_classes.items():
                if class_id in ids:
                    army_counts[group] += 1

            # Track explored tiles (rounded to tile level)
            if "x" in pos and "y" in pos:
                tile = (int(pos["x"]) // 4, int(pos["y"]) // 4)
                explored_tiles.add(tile)

        results[name] = {
            "army_composition": dict(army_counts),
            "explored_tiles": len(explored_tiles),
            "explored_pct": round((len(explored_tiles) / (128 * 128)) * 100, 2)  # 128x128 = map grid
        }

    return results

def build_strategy_vectors(army_data):
    vectors = []
    names = []

    for player, stats in army_data.items():
        comp = stats["army_composition"]
        vec = [
            comp.get("infantry", 0),
            comp.get("cavalry", 0),
            comp.get("archers", 0),
            comp.get("siege", 0),
            comp.get("monks", 0),
            stats["explored_pct"]
        ]
        vectors.append(vec)
        names.append(player)

    return np.array(vectors), names

def cluster_strategies(vectors, names, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(vectors)

    # Print clustered players
    for name, label in zip(names, labels):
        print(f"{name} → Strategy Cluster {label}")

    # Optional: visualize in 2D (PCA)
    from sklearn.decomposition import PCA
    reduced = PCA(n_components=2).fit_transform(vectors)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis')
    for i, name in enumerate(names):
        plt.annotate(name, (reduced[i, 0], reduced[i, 1]))
    plt.title("Player Strategies Clustered")
    plt.show()

    return labels

# Example usage:
# Step 1: Extract army + fog state from JSON
army_data = parse_army_state_vectors_and_fog("AOE2BOT/replays/Replay1.json")

# Step 2: Convert to numeric feature vectors
vectors, names = build_strategy_vectors(army_data)

# Step 3: Cluster players into strategies
k = min(3, len(vectors))  # Choose up to 3 clusters, but no more than number of players
cluster_labels = cluster_strategies(vectors, names, k)
