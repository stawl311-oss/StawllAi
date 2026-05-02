import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_network import StawllNet


def build_fake_dataloader(
    num_samples: int = 4096,
    input_dim: int = 128,
    output_dim: int = 10,
    batch_size: int = 64,
):
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = 128
    output_dim = 10

    model = StawllNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=512,
        depth=8,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    train_loader = build_fake_dataloader(
        num_samples=4096,
        input_dim=input_dim,
        output_dim=output_dim,
        batch_size=64,
    )

    epochs = 10
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}")


if __name__ == "__main__":
    train()
