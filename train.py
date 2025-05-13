from __future__ import annotations

import math
from pathlib import Path
from typing import List, Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wandb


# --------------------------------------------------------------------------- #
#                          Model architecture                                 #
# --------------------------------------------------------------------------- #
class StateEncoderCNN(nn.Module):
    """CNN that maps 4×104×68 tensors → state embedding."""

    def __init__(self, in_channels: int = 4, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # → 32×52×34
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),           # → 64×26×17
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # → 128×13×9
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 13 * 9, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 4, 104, 68)
        return self.net(x)


class ActionEncoderFC(nn.Module):
    """FC network mapping 7‑D action vector → action embedding."""

    def __init__(self, in_dim: int = 7, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:  # a: (B, 7)
        return self.net(a)


class QNetwork(nn.Module):
    """Full Q‑network Q(s, a) = f(φ_s(s), φ_a(a))."""

    def __init__(self,
                 state_embed_dim: int = 256,
                 action_embed_dim: int = 128,
                 hidden_dim: int = 512):
        super().__init__()
        self.state_encoder = StateEncoderCNN(out_dim=state_embed_dim)
        self.action_encoder = ActionEncoderFC(out_dim=action_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(state_embed_dim + action_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # scalar Q
        )

    def forward(self, state_tensor: torch.Tensor, action_vec: torch.Tensor) -> torch.Tensor:
        s_emb = self.state_encoder(state_tensor)
        a_emb = self.action_encoder(action_vec)
        x = torch.cat([s_emb, a_emb], dim=-1)
        return self.fc(x).squeeze(-1)  # (B,)


# --------------------------------------------------------------------------- #
#                                Dataset                                      #
# --------------------------------------------------------------------------- #
class SoccerDataset(Dataset):
    """Wraps preprocessed possession‑based trajectories.

    Each item corresponds to a single timestep within a trajectory.
    The data dict is expected to have the following keys:
        'state'      : FloatTensor 4×104×68
        'action'     : FloatTensor 7
        'reward'     : float
        'next_state' : FloatTensor 4×104×68
        'next_action': FloatTensor 7
        'done'       : bool (trajectory terminal flag)
        'traj_id'    : int  (identifier to group timesteps)
        't'          : int  (timestep index within trajectory)
    """

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


# --------------------------------------------------------------------------- #
#                      Target computation utilities                           #
# --------------------------------------------------------------------------- #
TargetType = Literal['mc', 'td0']


class TargetComputer:
    """Compute targets for batches based on chosen estimator."""

    def __init__(self,
                 gamma: float = 0.99,
                 target_type: TargetType = 'td_lambda'):
        self.gamma = gamma
        self.target_type = target_type

    def compute_targets(self,
                        batch: Dict[str, torch.Tensor],
                        q_net: QNetwork | None = None,
                        device: torch.device | str = 'cpu') -> torch.Tensor:
        """Return targets tensor shaped like (B,).

        For TD‑based targets, q_net must be provided.
        """
        rewards = batch['reward'].to(device)           # (B,)
        dones = batch['done'].to(device)               # (B,)
        if self.target_type == 'mc':
            return batch['G'].to(device)               # pre‑computed full returns

        elif self.target_type == 'td0':
            assert q_net is not None, "q_net required for TD‑targets"
            next_q = q_net(batch['next_state'].to(device),
                           batch['next_action'].to(device)).detach()
            targets = rewards + self.gamma * next_q * (~dones)
            return targets

        else:
            raise ValueError(f"Unknown target_type {self.target_type}")
        
        
# --------------------------------------------------------------------------- #
#                           Training / Evaluation                             #
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, q_net: QNetwork, dataset: torch.utils.data.Dataset,
                 batch_size: int = 64, lr: float = 1e-4, weight_decay: float = 1e-5,
                 epochs: int = 10, gamma: float = 0.99,
                 target_type: TargetType = 'td_lambda', log_interval: int = 100,
                 project: str = 'offline-soccer-rl'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = q_net.to(self.device)

        # iterable datasets cannot be shuffled by DataLoader
        is_iterable = isinstance(dataset, IterableDataset)
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=not is_iterable, num_workers=4,
                                 pin_memory=True, persistent_workers=True)  # type: ignore[arg-type]

        self.optim = optim.Adam(self.q_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.target_computer = TargetComputer(gamma, target_type)
        self.log_interval = log_interval
        self.step = 0

        self.run = wandb.init(project=project, config=dict(batch_size=batch_size, lr=lr,
                                                            epochs=epochs, gamma=gamma,
                                                            target_type=target_type))

    def train(self):
        self.q_net.train()
        for epoch in range(1, self.epochs + 1):
            for batch in self.loader:
                self.optim.zero_grad()
                preds = self.q_net(batch['state'].to(self.device), batch['action'].to(self.device))
                targets = self.target_computer.compute_targets(batch, self.q_net, self.device)
                loss = nn.functional.mse_loss(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.optim.step()

                if self.step % self.log_interval == 0:
                    wandb.log({'loss': loss.item(), 'epoch': epoch, 'step': self.step})
                self.step += 1
            print(f"[Epoch {epoch}/{self.epochs}] loss {loss.item():.4f}")

    def save(self, path: str | Path):
        torch.save(self.q_net.state_dict(), str(path))




import pickle, random, torch
from torch.utils.data import IterableDataset
from pathlib import Path

class ShardedSoccerDataset(IterableDataset):
    def __init__(self, shard_dir: str | Path, shuffle_shards=True):
        self.paths = sorted(Path(shard_dir).glob("*.pkl"))
        self.shuffle_shards = shuffle_shards

    def __iter__(self):
        paths = self.paths.copy()
        if self.shuffle_shards:
            random.shuffle(paths)        # order of shards each epoch
        for p in paths:
            with p.open("rb") as f:
                shard = pickle.load(f)   # load ONE shard into RAM
            random.shuffle(shard)        # in-shard shuffle
            for sample in shard:
                yield sample             # hand a single transition to DataLoader
                
# --------------------------------------------------------------------------- #
#                               Entry point                                   #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', default='./data/processed', type=Path,
                    help='Directory containing shard .pkl files')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--target', choices=['mc', 'td0'], default='td0')
    args = ap.parse_args()

    shard_paths = sorted(args.shards.glob('*.pkl'))
    dataset = ShardedSoccerDataset(shard_paths)

    q_net = QNetwork()
    trainer = Trainer(q_net, dataset, epochs=args.epochs, target_type=args.target)
    trainer.train()
    trainer.save('q_network.pt')
    wandb.finish()