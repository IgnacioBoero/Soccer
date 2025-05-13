from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

FIELD_X, FIELD_Y = 104, 68
ACTION_ID = {"pass": 0, "cross": 0,"bad_touch":1, "dribble": 1, "take_on": 1, "shot": 2}
N_ACTIONS = 3

# --------------------------------------------------------------------------- #
#                         Encoding helpers                                    #
# --------------------------------------------------------------------------- #

def encode_action(evt: Dict[str, Any]) -> np.ndarray:
    one_hot = np.zeros(N_ACTIONS, dtype=np.float32)
    aid = ACTION_ID[evt["action"]]
    one_hot[aid] = 1.0
    start = np.array(evt["ball_start"], dtype=np.float32)
    end = np.array(evt.get("ball_end", start), dtype=np.float32)
    return np.concatenate([one_hot, start, end])  # (7,)


def build_state(ball_start: List[float],
                players: List[Dict[str, Any]]) -> np.ndarray:
    """Return 4×104×68 float32 tensor."""
    tm = np.zeros((FIELD_X, FIELD_Y), dtype=np.float32)
    op = np.zeros((FIELD_X, FIELD_Y), dtype=np.float32)

    for pl in players:
        x, y = pl["location"]
        xi = int(np.clip(round(x), 0, FIELD_X - 1))
        yi = int(np.clip(round(y), 0, FIELD_Y - 1))
        (tm if pl["teammate"] else op)[xi, yi] = 1.0

    bx, by = ball_start
    xs = np.arange(FIELD_X).reshape(-1, 1)
    ys = np.arange(FIELD_Y).reshape(1, -1)
    dx, dy = xs - bx, ys - by
    dist = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    angle = np.arctan2(dy, dx, dtype=np.float32) / np.pi  # [-1,1]

    return np.stack([tm, op, dist, angle]).astype(np.float32)



# --------------------------------------------------------------------------- #
#                        Reward & returns                                     #
# --------------------------------------------------------------------------- #

def reward_fn(evt: Dict[str, Any]) -> float:
    # 100 if event is shot and outcome True
    # 10 if event is shot and outcome False but on_target is True
    # 5 if event is shot and outcome False but on_target is False
    # 0 elsewhere if ball finished y coordinate is grater than FIELD_Y / 2
    # -10 if ball finshed y coordinate is less than FIELD_Y / 2
    if evt["action"] == "shot":
        if evt["outcome"] == True:
            return 100
        elif evt["outcome"] == False and evt["on_target"] == True:
            return 10
        elif evt["outcome"] == False and evt["on_target"] == False:
            return 5
    elif evt["ball_end"][1] > FIELD_Y / 2:
        return 0
    elif evt["ball_end"][1] < FIELD_Y / 2:
        return -10
    return 0.0




def extract_pl_list(player_loc_entry: Any) -> Optional[List[Dict[str, Any]]]:
    """Return list of player dicts or None if missing/NaN."""
    
    try:
        players = player_loc_entry.values[0]
    except:
        players = None
    if isinstance(players,float):
        players = None
    return players  # type: ignore

def resolve_player_lists(raw_traj: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Forward/backward fill missing player locations in a trajectory."""
    raw_lists: List[Optional[List[Dict[str, Any]]]] = [
        extract_pl_list(evt.get("player_loc")) for evt in raw_traj
    ]

    # forward fill ------------------------------------------------------- #
    last_valid: Optional[List[Dict[str, Any]]] = None
    for i, lst in enumerate(raw_lists):
        if lst is None and last_valid is not None:
            raw_lists[i] = last_valid
        elif lst is not None:
            last_valid = lst

    # backward fill for prefix Nones ------------------------------------ #
    next_valid: Optional[List[Dict[str, Any]]] = None
    for i in reversed(range(len(raw_lists))):
        if raw_lists[i] is None and next_valid is not None:
            raw_lists[i] = next_valid
        elif raw_lists[i] is not None:
            next_valid = raw_lists[i]

    if any(lst is None for lst in raw_lists):
        print("WARNING: Some or all player lists might be missing. Replacing with empty lists.")
        empty: List[Dict[str, Any]] = []
        raw_lists = [lst or empty for lst in raw_lists]
    return raw_lists




def discount_cumsum(rs: List[float], gamma: float) -> List[float]:
    G, out = 0.0, [0.0] * len(rs)
    for t in reversed(range(len(rs))):
        G = rs[t] + gamma * G
        out[t] = G
    return out

# --------------------------------------------------------------------------- #
#                       Trajectory processing                                 #
# --------------------------------------------------------------------------- #
def process_single_traj(traj: List[Dict[str, Any]],
                        traj_id: int,
                        gamma: float) -> List[Dict[str, Any]]:
    players_seq = resolve_player_lists(traj)
    steps: List[Dict[str, Any]] = []

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

    # link next_state / next_action -------------------------------------- #
    for i in range(len(steps) - 1):
        steps[i]["next_state"] = steps[i + 1]["state"]
        steps[i]["next_action"] = steps[i + 1]["action"]
    steps[-1]["done"] = True

    # reward only at terminal step --------------------------------------- #
    steps[-1]["reward"] = reward_fn(traj[-1])
    
    # Monte‑Carlo returns
    Gs = discount_cumsum([s["reward"] for s in steps], gamma)
    for s, G in zip(steps, Gs):
        s["G"] = G
    return steps



# --------------------------------------------------------------------------- #
#                       Shard writer + CLI                                   #
# --------------------------------------------------------------------------- #

def write_shard(buf: List[List[Dict[str, Any]]], idx: int, out_dir: Path):
    """Flatten trajectory lists into one list of steps and pickle it."""
    flat = [s for traj in buf for s in traj]
    out_path = out_dir / f"trajectories_{idx:03d}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(flat, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  • wrote {len(flat)} transitions from {len(buf)} trajectories → {out_path}")
    
def cli():
    ap = argparse.ArgumentParser("Prepare soccer data (sharded, robust)")
    ap.add_argument("--input", "-i", type=Path, default='data/trajectories_left2right.pkl')
    ap.add_argument("--outdir", "-o", type=Path, default='data/processed')
    ap.add_argument("--shard-size", "-n", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.9)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "rb") as f:
        raw_data = pickle.load(f)

    buf: List[List[Dict[str, Any]]] = []
    shard_idx = 0
    for traj_id, traj in enumerate(raw_data):
        buf.append(process_single_traj(traj, traj_id, args.gamma))
        if len(buf) >= args.shard_size:
            write_shard(buf, shard_idx, args.outdir)
            buf.clear(); shard_idx += 1

    if buf:
        write_shard(buf, shard_idx, args.outdir)

    print("✅  All shards written.")


if __name__ == "__main__":
    cli()