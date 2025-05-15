## CODE TO PREPROCESS THE DATA
import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import random

# HYPERPARAMETERS
FIELD_X, FIELD_Y = 104, 68
ACTION_ID = {"pass": 0, "cross": 0,"bad_touch":1, "dribble": 1, "take_on": 1, "shot": 2}
N_ACTIONS = 3


## FUNCTIONS TO GET THE ACTION AND STATE FROM THE STAT BOMB RAW DATA

def encode_action(evt: Dict[str, Any]) -> np.ndarray:
    # One hot of type of action
    one_hot = np.zeros(N_ACTIONS, dtype=np.float32)
    aid = ACTION_ID[evt["action"]]
    one_hot[aid] = 1.0
    # ball start and end
    start = np.array(evt["ball_start"], dtype=np.float32)
    end = np.array(evt.get("ball_end", start), dtype=np.float32)
    return np.concatenate([one_hot, start, end])  # (7,)


def build_state(ball_start: List[float],
                players: List[Dict[str, Any]]) -> np.ndarray:
    # Matrices for players positons
    tm = np.zeros((FIELD_X, FIELD_Y), dtype=np.float32)
    op = np.zeros((FIELD_X, FIELD_Y), dtype=np.float32)
    # Set to 1 where there are players  
    for pl in players:
        x, y = pl["location"]
        xi = int(np.clip(round(x), 0, FIELD_X - 1))
        yi = int(np.clip(round(y), 0, FIELD_Y - 1))
        (tm if pl["teammate"] else op)[xi, yi] = 1.0
    # Matrixes for ball position (dist anda ngle)
    bx, by = ball_start
    xs = np.arange(FIELD_X).reshape(-1, 1)
    ys = np.arange(FIELD_Y).reshape(1, -1)
    dx, dy = xs - bx, ys - by
    dist = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    angle = np.arctan2(dy, dx, dtype=np.float32) / np.pi
    # Return the 4 concat matricse as state
    return np.stack([tm, op, dist, angle]).astype(np.float32)


## Get reward of the last action
def reward_fn(evt: Dict[str, Any]) -> float:
    # 100 if event is shot and  goal
    # 20 if event is shot and miss but on target
    # 10 if event is shot and miss and off target
    # -10 if ball lost in other teams half
    # -20 if ball lost in own court
    if evt["action"] == "shot":
        if evt["outcome"] == True:
            return 100
        elif evt["outcome"] == False and evt["on_target"] == True:
            return 20
        elif evt["outcome"] == False and evt["on_target"] == False:
            return 10
    elif evt["ball_end"][1] > FIELD_Y / 2:
        return -10
    elif evt["ball_end"][1] < FIELD_Y / 2:
        return -20
    return 0.0



## AUXILIARY FUNCTIONS TO GET PLAYERS POSITIONS

def extract_pl_list(player_loc_entry: Any) -> Optional[List[Dict[str, Any]]]:    
    try:
        players = player_loc_entry.values[0]
    except:
        players = None
    if isinstance(players,float):
        players = None
    return players

def resolve_player_lists(raw_traj: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    # Some events had missing data of players positions, so they are filled with the last valid position from the same trajectory
    raw_lists: List[Optional[List[Dict[str, Any]]]] = [
        extract_pl_list(evt.get("player_loc")) for evt in raw_traj
    ]

    last_valid: Optional[List[Dict[str, Any]]] = None
    for i, lst in enumerate(raw_lists):
        if lst is None and last_valid is not None:
            raw_lists[i] = last_valid
        elif lst is not None:
            last_valid = lst

    next_valid: Optional[List[Dict[str, Any]]] = None
    for i in reversed(range(len(raw_lists))):
        if raw_lists[i] is None and next_valid is not None:
            raw_lists[i] = next_valid
        elif raw_lists[i] is not None:
            next_valid = raw_lists[i]
    # If no event in the trajectory has palers data, discard
    if any(lst is None for lst in raw_lists):
        empty: List[Dict[str, Any]] = []
        raw_lists = [lst or empty for lst in raw_lists]
        return None
    return raw_lists



## AUX FUNCTION FOR THE RETURN OF EACH STATE
def discount_cumsum(rs: List[float], gamma: float) -> List[float]:
    G, out = 0.0, [0.0] * len(rs)
    for t in reversed(range(len(rs))):
        G = rs[t] + gamma * G
        out[t] = G
    return out



## PROCESS EACH TRAJECTORY
def process_single_traj(traj: List[Dict[str, Any]],
                        traj_id: int,
                        gamma: float) -> List[Dict[str, Any]]:
    
    # Get players positions
    players_seq = resolve_player_lists(traj)
    if not isinstance(players_seq,list):
        print('Players info not available, dropping out trajectoru')
        return None
    steps: List[Dict[str, Any]] = []

    # Initialize the event values
    for t, (evt, players) in enumerate(zip(traj, players_seq)):
        state_arr = build_state(evt["ball_start"], players)
        step = {
            "state": torch.tensor(state_arr),
            "action": torch.tensor(encode_action(evt)),
            "reward": 0.0,  # placeholder
            "next_state": torch.zeros((4, 104, 68), dtype=torch.float32),
            "next_action": torch.zeros(7, dtype=torch.float32),
            "done": False,
            "traj_id": traj_id,
            "t": t,
        }
        steps.append(step)

    # fill next state and next action
    for i in range(len(steps) - 1):
        steps[i]["next_state"] = steps[i + 1]["state"]
        steps[i]["next_action"] = steps[i + 1]["action"]
    steps[-1]["done"] = True

    # fill reward at last event
    steps[-1]["reward"] = reward_fn(traj[-1])
    
    # Fill return for all event
    Gs = discount_cumsum([s["reward"] for s in steps], gamma)
    for s, G in zip(steps, Gs):
        s["G"] = G
    return steps





# AUXILIAR FUNCTION THAT WRITES SHARDS (BECAUSE THAT IS TOO BIG, WRITE SHARDS OF 1000 TRAJS)
def write_shard(buf: List[List[Dict[str, Any]]], idx: int, out_dir: Path):

    # Mix trajectories
    flat = [s for traj in buf for s in traj]
    random.shuffle(flat)
    
    # Split train and val
    split_idx = int(0.8 * len(flat))
    train_data = flat[:split_idx]
    val_data = flat[split_idx:]

    # Save
    out_path_train = out_dir / "train" / f"trajectories_{idx:03d}.pkl"
    out_path_val = out_dir / "val" / f"trajectories_{idx:03d}.pkl"

    out_path_train.parent.mkdir(parents=True, exist_ok=True)
    out_path_val.parent.mkdir(parents=True, exist_ok=True)

    with out_path_train.open("wb") as f:
        pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with out_path_val.open("wb") as f:
        pickle.dump(val_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"wrote {len(train_data)} to {out_path_train} and {len(val_data)} to {out_path_val}")
    
def main():
    ap = argparse.ArgumentParser("Prepare soccer data (sharded, robust)")
    ap.add_argument("--input", "-i", type=Path, default='data/trajectories_left2right.pkl')
    ap.add_argument("--outdir", "-o", type=Path, default='data/processed')
    ap.add_argument("--shard-size", "-n", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.99)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    with open(args.input, "rb") as f:
        raw_data = pickle.load(f)

    buf: List[List[Dict[str, Any]]] = []
    shard_idx = 0
    dropped_tr = 0
    
    # Process trajs
    for traj_id, traj in enumerate(raw_data):
        processed = process_single_traj(traj, traj_id, args.gamma)
        # Discard trajs where players info is missing
        if isinstance(processed, list):
            buf.append(processed)
        else:
            dropped_tr += 1
        # Save every 1000 trajs
        if len(buf) >= args.shard_size:
            print(f'{dropped_tr}/{len(buf)} dropped trajectories.')
            dropped_tr = 0
            write_shard(buf, shard_idx, args.outdir)
            buf.clear(); shard_idx += 1
    # Save remaining of last shard
    if buf:
        write_shard(buf, shard_idx, args.outdir)

    print("All shards written.")


if __name__ == "__main__":
    main()