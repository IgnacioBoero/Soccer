import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

import copy



# Normalization utils
@dataclass
class Stats:
    state_mean: torch.Tensor  # (4,)
    state_std: torch.Tensor   # (4,)
    action_mean: torch.Tensor  # (7,)
    action_std: torch.Tensor   # (7,)
    reward_mean: float
    reward_std: float
    G_mean: float
    G_std: float


class Normaliser:
    # Norm stats
    def __init__(self, stats: Stats, device: torch.device | str = "cpu"):
        self.state_mean = stats.state_mean.to(device).view(4, 1, 1)
        self.state_std = stats.state_std.to(device).view(4, 1, 1)
        self.action_mean = stats.action_mean.to(device)
        self.action_std = stats.action_std.to(device)
        self.reward_mean = torch.tensor(stats.reward_mean, device=device)
        self.reward_std = torch.tensor(stats.reward_std, device=device)
        self.G_mean = torch.tensor(stats.G_mean, device=device)
        self.G_std = torch.tensor(stats.G_std, device=device)

    # Execute normalization
    def state(self, x: torch.Tensor) -> torch.Tensor:
        #return (x - self.state_mean) / self.state_std
        return x
    def action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def rewards(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.reward_mean) / self.reward_std
    
    def G(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.G_mean) / self.G_std




## STATE ENCODER ARQ
class StateEncoderCNN(nn.Module):
    def __init__(self, in_channels: int = 4, out_dim: int = 256, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 3 * 2, out_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

## ACTION ENCODER ARQ
class ActionEncoderFC(nn.Module):
    def __init__(self, in_dim: int = 7, out_dim: int = 128, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)

## Q NET ARQ
class QNetwork(nn.Module):
    def __init__(
        self,
        hidden_layers: Sequence[int] | None = None,
        state_embed_dim: int = 128,
        action_embed_dim: int = 128,
        p_drop=0.0,
    ) -> None:
        super().__init__()
        hidden_layers = list(hidden_layers or [128])
        self.state_encoder = StateEncoderCNN(out_dim=state_embed_dim, p_drop=p_drop)
        self.action_encoder = ActionEncoderFC(out_dim=action_embed_dim, p_drop=p_drop)

        layers: List[nn.Module] = []
        in_dim = state_embed_dim + action_embed_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(inplace=True)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, state_tensor: torch.Tensor, action_vec: torch.Tensor) -> torch.Tensor:
        s_emb = self.state_encoder(state_tensor)
        a_emb = self.action_encoder(action_vec)
        x = torch.cat([s_emb, a_emb], dim=-1)
        return self.fc(x).squeeze(-1)




## AUX FUNCTION TO GET THE TARGET (MC or TD0)

TargetType = Literal["mc", "td0"]
class TargetComputer:
    def __init__(
        self,
        gamma: float,
        target_type: TargetType,
        normaliser: Normaliser | None = None,
    ):
        self.gamma = gamma
        self.target_type = target_type
        self.norm = normaliser
    def compute(self, batch: Dict[str, torch.Tensor], q_net: QNetwork, device: torch.device):
        rewards = batch["reward"].to(device)
        dones = batch["done"].to(device)
        Gs = batch["G"].to(device)
        if self.target_type == "mc":
            raw_targets = Gs
        elif self.target_type == "td0":
            next_q = q_net(batch["next_state"].to(device), batch["next_action"].to(device)).detach()
            raw_targets = rewards + self.gamma * next_q * (~dones)
        else:
            raise ValueError(self.target_type)
        return (raw_targets - self.norm.G_mean) / self.norm.G_std



## ITERABLE DATASET FOR SMALL SHARDS
class ShardedSoccerDataset(IterableDataset):
    def __init__(self, shard_paths: Sequence[Path], shuffle_shards: bool = True):
        self.paths = list(shard_paths)
        self.shuffle = shuffle_shards
    def __iter__(self):
        paths = self.paths.copy()
        if self.shuffle:
            random.shuffle(paths)
        for p in paths:
            with p.open("rb") as f:
                shard = pickle.load(f)
            if self.shuffle:
                random.shuffle(shard)
            for sample in shard:
                yield sample


## GET NORMALIZATION VALUES FROM TRAINING DATA
def compute_stats(shard_paths: Sequence[Path]) -> Stats:

    state_sum = torch.zeros(4)
    state_sq = torch.zeros(4)
    action_sum = torch.zeros(7)
    action_sq = torch.zeros(7)
    reward_sum = 0.0
    reward_sq = 0.0
    G_sum = 0.0
    G_sq = 0.0
    n_state_pix = 0
    n_samples = 0
    
    for p in tqdm(shard_paths, desc=f"Computing stats"):
        with p.open("rb") as f:
            shard = pickle.load(f)
        for s in shard:
            state = torch.as_tensor(s["state"], dtype=torch.float32)
            action = torch.as_tensor(s["action"], dtype=torch.float32)
            reward = float(s["reward"])
            G = float(s["G"])
            ch_means = state.view(4, -1).mean(dim=1)
            ch_vars = state.view(4, -1).var(dim=1, unbiased=False)
            state_sum += ch_means
            state_sq += ch_vars + ch_means ** 2

            action_sum += action
            action_sq += action ** 2

            reward_sum += reward
            reward_sq += reward ** 2

            G_sum += G
            G_sq += G ** 2

            n_state_pix += 1
            n_samples += 1

    state_mean = state_sum / n_state_pix
    state_var = state_sq / n_state_pix - state_mean ** 2
    state_std = torch.sqrt(torch.clamp(state_var, min=1e-8))

    action_mean = action_sum / n_samples
    action_var = action_sq / n_samples - action_mean ** 2
    action_std = torch.sqrt(torch.clamp(action_var, min=1e-8))

    reward_mean = reward_sum / n_samples
    reward_var = reward_sq / n_samples - reward_mean ** 2
    reward_std = math.sqrt(max(reward_var, 1e-8))

    G_mean = G_sum / n_samples
    G_var = G_sq / n_samples - G_mean ** 2
    G_std = math.sqrt(max(G_var, 1e-8))

    return Stats(
        state_mean, state_std,
        action_mean, action_std,
        reward_mean, reward_std,
        G_mean, G_std
    )


## AUX FUNCTION TO GET TRAIN AND VAL PATHS SHARDS
def make_train_val(shards_dir: Path):
    train_path = shards_dir / "train"
    train_paths = sorted(train_path.glob("*.pkl"))
    val_path = shards_dir / "val"
    val_paths = sorted(val_path.glob("*.pkl"))
    return train_paths, val_paths


## TRAINER
class Trainer:
    def __init__(
        self,
        q_net: QNetwork,
        train_ds: IterableDataset,
        val_ds: IterableDataset,
        normaliser: Normaliser,
        device,
        *,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        epochs: int = 10,
        gamma: float = 0.99,
        target_type: TargetType = "td0",
        log_interval: int = 200,
        project: str = "offline-soccer-rl",
        wandb_mode: str = "online",

    ):
        self.device = device
        self.q_net = q_net.to(self.device)
        self.q_target = copy.deepcopy(self.q_net).eval()  # frozen target
        self.norm = normaliser
        self.target_type = target_type

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        self.optim = optim.Adam(
            self.q_net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.epochs = epochs
        self.target_computer = TargetComputer(gamma, target_type, normaliser)
        self.log_interval = log_interval
        self.gstep = 0

        self.run = wandb.init(
            project=project,
            mode=wandb_mode,
            config=dict(
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                gamma=gamma,
                target_type=target_type,
            ),
        )

    
    # AUX FUNCTION TO NORMALIZE ALL ENTRIES OF THE BATCH
    def _normalise_batch(self, batch: Dict[str, torch.Tensor]):
        batch["state"] = self.norm.state(batch["state"].to(self.device))
        batch["next_state"] = self.norm.state(batch["next_state"].to(self.device))
        batch["action"] = self.norm.action(batch["action"].to(self.device))
        batch["next_action"] = self.norm.action(batch["next_action"].to(self.device))
        batch["reward"] = (
            torch.as_tensor(batch["reward"], device=self.device, dtype=torch.float32)
        )
        batch["G"] = (
            torch.as_tensor(batch["G"], device=self.device, dtype=torch.float32)
        )
        batch["done"] = torch.as_tensor(batch["done"], device=self.device, dtype=torch.bool)
        return batch

    # RUN ONE EPOCH
    def _epoch(self, loader: DataLoader, train: bool):
        self.q_net.train(train)
        mode = "train" if train else "val"
        total_loss = 0.0
        samples = 0
        pbar = tqdm(loader, desc=f"{mode.title()} {self.current_epoch}", leave=False)
        for batch_raw in pbar:
            batch = self._normalise_batch(batch_raw)
            preds = self.q_net(batch["state"], batch["action"])
            targets = self.target_computer.compute(batch, self.q_target, self.device)
            loss = nn.functional.mse_loss(preds, targets)

            bs = preds.size(0)
            total_loss += loss.item() * bs
            samples += bs

            if train:
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.optim.step()
                if self.gstep % 2000 == 0:
                    self.q_target.load_state_dict(self.q_net.state_dict())
                if self.gstep % self.log_interval == 0:
                    wandb.log({f"{mode}/batch_loss": loss.item(), "step": self.gstep})
                self.gstep += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(samples, 1)

    ## MAIN TRAIN FUNCTION
    def train(self):
        best_val, patience = float("inf"), 10
        # Iterate over epochs
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            tr_loss = self._epoch(self.train_loader, True)
            val_loss = self._epoch(self.val_loader, False)
            wandb.log({"epoch": epoch, "train/loss": tr_loss, "val/loss": val_loss})
            print(
                f"[Epoch {epoch}/{self.epochs}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                patience = 3
                self.save(f"models/{self.target_type}/best.pt")
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping")
                    break
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()
        self.save(f"models/{self.target_type}/final.pt")
        wandb.log({"final/val_loss": self.evaluate()})
        wandb.finish()

    # EVAL AUX
    def evaluate(self):
        return self._epoch(self.val_loader, False)

    # SAVE WEITGHTS AND NORMALIZER VALUES
    def save(self, path: Path | str):
            norm_state = {
                "state_mean": self.norm.state_mean.cpu(),
                "state_std":  self.norm.state_std.cpu(),
                "action_mean": self.norm.action_mean.cpu(),
                "action_std":  self.norm.action_std.cpu(),
                "reward_mean": float(self.norm.reward_mean),
                "reward_std":  float(self.norm.reward_std),
                "G_mean":      float(self.norm.G_mean),
                "G_std":       float(self.norm.G_std),
            }

            checkpoint = {
                "q_net": self.q_net.state_dict(),
                "optimizer": self.optim.state_dict(),
                "normaliser": norm_state,
            }
            torch.save(checkpoint, str(path))




## OPTUNA HYPERPARAMETERS SEARCH

def _objective(trial: optuna.Trial, train_paths: List[Path], val_paths: List[Path], epochs: int, target_type: str, device):
    # hyper‑parameters
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_layers = trial.suggest_int("n_layers", 3, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [512])
    hidden_layers = [hidden_dim] * n_layers
    p_drop = trial.suggest_float("p_drop", 0.0, 0.1, step=0.05)

    # Get norm stats
    stats = compute_stats(train_paths)
    norm = Normaliser(stats, device)

    # Train
    q_net = QNetwork(hidden_layers=hidden_layers,p_drop=p_drop)
    trainer = Trainer(
        q_net,
        ShardedSoccerDataset(train_paths, True),
        ShardedSoccerDataset(val_paths, False),
        norm,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        target_type=target_type,
        project=f"offline-soccer-rl-{target_type}",
        device=device
    )
    trainer.train()
    trainer.save(
        f"models/{target_type}/optuna_lr{lr}_hd{hidden_dim}_nl{n_layers}.pt"
    )
    return trainer.evaluate()


## OPTUNA MAIN FUNCTION
def main(shards_dir: Path, trials: int, epochs: int, target_type: str, device):
    train_paths, val_paths = make_train_val(shards_dir)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: _objective(t, train_paths, val_paths, epochs, target_type,device),
        n_trials=trials,
    )
    print(f"Best trial {study.best_trial.number}: {study.best_value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")



if __name__ == "__main__":
    import argparse
    random.seed(42)
    torch.manual_seed(42)
    # Parse args
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default="./data/processed")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--target", choices=["mc", "td0"], default="td0")
    ap.add_argument("--trials" ,type=int, default=64)
    ap.add_argument("--device", type=str, default='cuda:0')
    args = ap.parse_args()
    # Train
    main(args.shards, args.trials, args.epochs, args.target, args.device)

