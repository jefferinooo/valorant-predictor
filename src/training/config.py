from dataclasses import dataclass


@dataclass
class TrainConfig:
    input_dim: int = 15
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    val_split: float = 0.15
    seed: int = 42
    checkpoint_path: str = "artifacts/models/best.pt"
