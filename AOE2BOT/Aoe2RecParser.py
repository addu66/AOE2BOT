# import json
# import mgz

# def parse_aoe2record_to_json(file_path):
#     """Parses an AoE2 recorded game file and converts it to JSON format."""
#     with open(file_path, 'rb') as file:
#         parser = mgz.Parser(file)
#         header = parser.parse_header()
#         metadata = parser.metadata
#         # Parse additional details as needed

#     game_data = {
#         "version": header.version,
#         "map": metadata.map_name,
#         "players": [
#             {
#                 "name": player.name,
#                 "civ": player.civilization,
#                 "team": player.team
#             } for player in metadata.players
#         ],
#         # Add more extracted data as needed
#     }

#     json_file = file_path.replace(".aoe2record", ".json")
#     with open(json_file, "w") as json_out:
#         json.dump(game_data, json_out, indent=4)

#     print(f"Converted {file_path} to {json_file}")
#     return json_file

# # Example Usage
# parse_aoe2record_to_json("AOE2BOT/replays/Replay1.aoe2record")


import json
from mgz.model import parse_match, serialize
import pandas as pd

# file_path = 'AOE2BOT/replays/Replay1.aoe2record'

# with open(file_path, 'rb') as h:
#     match = parse_match(h)
#     # print(json.dumps(serialize(match), indent=2))
#     json_file = file_path.replace(".aoe2record", ".json")
#     with open(json_file, "w") as json_out:
#         json.dump(serialize(match), json_out, indent=4)



def load_replay(file_path):
    """ Load parsed AoE2 replay data """
    with open(file_path, 'r') as f:
        return json.load(f)

def process_frames(game_data):
    """ Extract frame-by-frame data from replay """
    frames = []
    
    for match in game_data:
        for frame in match['frames']:  # Assuming 'frames' contains all game states
            for player in frame['players']:
                observable_map = frame['map'][player['id']]  # Player's visible area
                actions = player.get('actions', [])  # Actions in this frame
                
                frames.append({
                    "time": frame['time'],
                    "player_id": player['id'],
                    "food": player['resources']['food'],
                    "wood": player['resources']['wood'],
                    "gold": player['resources']['gold'],
                    "stone": player['resources']['stone'],
                    "units": player['military']['unit_count'],
                    "villagers": player['economy']['villager_count'],
                    "buildings": player['buildings']['building_count'],
                    "age": player['age'],
                    "observable_map": observable_map,  # Fog of war applied
                    "actions": actions  # What the player did this frame
                })

    return pd.DataFrame(frames)

# # Load replay data
game_records = load_replay("AOE2BOT/replays/Replay1.json")
# df = process_frames(game_records)

# # Save to CSV for analysis
# df.to_csv("frame_by_frame_data.csv", index=False)
# print("Frame-by-frame data saved.")

from collections import Counter
import json

def get_army_composition(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    military_units = set([
        "Militia", "Man-at-Arms", "Archer", "Skirmisher", "Scout Cavalry", 
        "Knight", "Camel", "Spearman", "Eagle", "Longbowman", "Mangonel",
        "Crossbowman", "Chu Ko Nu", "Elephant", "Samurai"
        # Add more AoE2 military units as needed
    ])

    army_composition = {}

    for player in data["players"]:
        player_name = player["name"]
        unit_counts = Counter()

        for obj in player.get("objects", []):
            unit_name = obj.get("name", "")
            if unit_name in military_units:
                unit_counts[unit_name] += 1

        army_composition[player_name] = dict(unit_counts)

    return army_composition

print(get_army_composition("AOE2BOT/replays/Replay1.json"))
