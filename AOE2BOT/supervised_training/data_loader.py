import json
import os

def load_replay_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_all_replays(replays_dir='../replays/'):
    replays = []
    for file in os.listdir(replays_dir):
        if file.endswith('.json'):
            replays.append(load_replay_json(os.path.join(replays_dir, file)))
    return replays