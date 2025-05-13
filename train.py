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
TargetType = Literal['mc', 'td0', 'td_lambda']


class TargetComputer:
    """Compute targets for batches based on chosen estimator."""

    def __init__(self,
                 gamma: float = 0.99,
                 lambda_: float = 0.8,
                 target_type: TargetType = 'td_lambda'):
        self.gamma = gamma
        self.lambda_ = lambda_
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

        elif self.target_type == 'td_lambda':
            assert 'lambda_return' in batch, "λ‑returns must be pre‑computed per sample"
            return batch['lambda_return'].to(device)

        else:
            raise ValueError(f"Unknown target_type {self.target_type}")
# --------------------------------------------------------------------------- #
#                           Training / Evaluation                             #
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self,
                 q_net: QNetwork,
                 dataset: SoccerDataset,
                 batch_size: int = 64,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 epochs: int = 10,
                 gamma: float = 0.99,
                 lambda_: float = 0.8,
                 target_type: TargetType = 'td_lambda',
                 log_interval: int = 100,
                 project: str = 'offline‑soccer‑rl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = q_net.to(self.device)
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, pin_memory=True)
        self.optim = optim.Adam(self.q_net.parameters(), lr=lr,
                                weight_decay=weight_decay)
        self.epochs = epochs
        self.target_computer = TargetComputer(gamma, lambda_, target_type)
        self.log_interval = log_interval
        self.step = 0

        # W&B init
        self.run = wandb.init(project=project, config={
            'batch_size': batch_size,
            'lr': lr,
            'epochs': epochs,
            'gamma': gamma,
            'lambda': lambda_,
            'target_type': target_type,
            'architecture': 'cnn+fc'
        })

    def train(self) -> None:
        self.q_net.train()
        for epoch in range(1, self.epochs + 1):
            for batch_idx, batch in enumerate(self.loader, start=1):
                self.optim.zero_grad()
                states = batch['state'].to(self.device)
                actions = batch['action'].to(self.device)

                preds = self.q_net(states, actions)
                targets = self.target_computer.compute_targets(batch,
                                                               q_net=self.q_net,
                                                               device=self.device)
                loss = nn.functional.mse_loss(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.optim.step()

                # Logging
                if self.step % self.log_interval == 0:
                    wandb.log({'loss': loss.item(),
                               'epoch': epoch,
                               'step': self.step})
                self.step += 1

            print(f"[Epoch {epoch}/{self.epochs}] Loss: {loss.item():.4f}")

    def save(self, path: str | Path) -> None:
        torch.save(self.q_net.state_dict(), Path(path))


# --------------------------------------------------------------------------- #
#                      Utilities to compute returns                           #
# --------------------------------------------------------------------------- #
def discount_cumsum(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted cumulative sums G_t for a trajectory."""
    G = 0.0
    out = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def lambda_returns(mc_returns: List[float], gamma: float, lambda_: float) -> List[float]:
    """Compute λ‑returns from MC returns in one pass (backwards)."""
    G_lambda = 0.0
    out = [0.0] * len(mc_returns)
    for t in reversed(range(len(mc_returns))):
        G_lambda = mc_returns[t] + gamma * lambda_ * G_lambda
        out[t] = G_lambda
    return out


def preprocess_trajectories(trajectories: List[List[Dict]],
                            gamma: float = 0.99,
                            lambda_: float = 0.8) -> List[Dict]:
    """Add MC and λ‑returns to each sample for faster training."""
    processed: List[Dict] = []
    for traj in trajectories:
        rewards = [step['reward'] for step in traj]
        mc = discount_cumsum(rewards, gamma)
        lam = lambda_returns(mc, gamma, lambda_)
        for i, step in enumerate(traj):
            step = step.copy()  # shallow copy
            step['G'] = mc[i]
            step['lambda_return'] = lam[i]
            processed.append(step)
    return processed


# --------------------------------------------------------------------------- #
#                               Entry point                                   #
# --------------------------------------------------------------------------- #
def main(data_path: str,
         epochs: int = 20,
         target_type: TargetType = 'td_0'):
    """Example training script."""
    # --------------------------------------------------------------------- #
    # Load your pre‑processed trajectories here. Replace with actual loader.
    # Expects a JSONL or pickle list[List[dict]] according to the spec above.
    import pickle
    trajectories: List[List[Dict]]
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    # --------------------------------------------------------------------- #
    trajectories = preprocess_trajectories(trajectories)

    dataset = SoccerDataset(trajectories)
    q_net = QNetwork()
    trainer = Trainer(q_net, dataset,
                      epochs=epochs,
                      target_type=target_type)
    trainer.train()
    trainer.save('q_network.pt')
    wandb.finish()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to offline trajectories (pickle)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--target', type=str, choices=['mc', 'td0'],
                        default='td_lambda')
    args = parser.parse_args()
    main(args.data, epochs=args.epochs, target_type=args.target)
