from mgz import header, body, fast
from mgz.model import parse_match, serialize
import json
from mgz.summary import Summary


def parse_file_full(path):
    with open(path, 'rb') as f:
        match = parse_match(f)
        json_file = path.replace(".aoe2record", ".json")
        with open(json_file, "w") as json_out:
            json.dump(serialize(match), json_out, indent=2)

def parse_file_summary(path, summary_class):
    with open(path, 'rb') as f:
        match = summary_class(f)
        print(match)
        json_file = path.replace(".aoe2record", ".json")
        with open(json_file, "w") as json_out:
            json.dump((match), json_out, indent=4)

def test_summary(filepath):
    with open(filepath, 'rb') as data:
        s = Summary(data)
        print(s.get_map())
        print(s.get_platform())
        # ... etc

rec = "AOE2BOT/replays/Replay2.aoe2record"
parse_file_full(rec)
# parse_file_summary(rec, Summary)
# test_summary(rec)
