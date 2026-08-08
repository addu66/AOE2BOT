import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def extract_features_and_labels(replay, action_vocab):
    sequences = []
    for player in replay['players']:  # Include all players, gaia not present
        player_num = player['number']
        # Initial state features: unit counts by class_id (assuming class_id up to 100)
        unit_counts = np.zeros(100)
        for obj in player['objects']:
            if obj['class_id'] < 100:
                unit_counts[obj['class_id']] += 1
        state_vec = unit_counts.tolist()
        
        # Actions for this player
        player_actions = [inp for inp in replay['inputs'] if 'player' in inp and inp['player'] == player_num]
        if len(player_actions) < 2:
            continue  # Need at least 2 actions for sequence
        for i in range(len(player_actions) - 1):
            # Sequence of previous actions
            seq_actions = []
            for a in player_actions[:i+1]:
                if 'type' in a:
                    seq_actions.append(a['type'] + '_' + str(a.get('param', '')))
            # Next action as label
            next_action = ''
            if 'type' in player_actions[i+1]:
                next_action = player_actions[i+1]['type'] + '_' + str(player_actions[i+1].get('param', ''))
            
            if seq_actions and next_action:
                # Convert to indices
                seq_indices = [action_vocab.get(act, action_vocab['<UNK>']) for act in seq_actions]
                label_index = action_vocab.get(next_action, action_vocab['<UNK>'])
                
                sequences.append({
                    'state': state_vec,
                    'seq': seq_indices,
                    'label': label_index
                })
    return sequences

def build_vocab(replays):
    all_actions = set()
    for replay in replays:
        for inp in replay['inputs']:
            if 'type' in inp:
                action_str = inp['type'] + '_' + str(inp.get('param', ''))
                all_actions.add(action_str)
    action_list = list(all_actions) + ['<UNK>']
    action_vocab = {act: idx for idx, act in enumerate(action_list)}
    return action_vocab

def preprocess_all(replays):
    action_vocab = build_vocab(replays)
    all_data = []
    for replay in replays:
        all_data.extend(extract_features_and_labels(replay, action_vocab))
    return all_data, action_vocab