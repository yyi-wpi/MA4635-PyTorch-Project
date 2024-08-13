import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import get_data_loaders, TrojanDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
import json
from datetime import datetime


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class AdvancedTransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.dropout(x)
        return self.fc(x)


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / total


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[
    float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation"):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / total


def plot_training_history(train_losses: List[float], train_accuracies: List[float], val_losses: List[float],
                          val_accuracies: List[float]):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, train_losses: List[float],
                    train_accuracies: List[float], val_losses: List[float], val_accuracies: List[float], filename: str):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filename: str) -> Tuple[
    int, List[float], List[float], List[float], List[float]]:
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    train_accuracies = checkpoint['train_accuracies']
    val_losses = checkpoint['val_losses']
    val_accuracies = checkpoint['val_accuracies']
    return epoch, train_losses, train_accuracies, val_losses, val_accuracies


def main():
    # Hyperparameters
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    dropout = 0.2

    # Load data using the existing utility function
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Get input dimension from the first batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]

    # Model parameters
    d_model = 128
    num_heads = 8
    num_layers = 4
    d_ff = 256
    num_classes = 2  # Binary classification for cybersecurity (e.g., malicious/benign)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedTransformerModel(input_dim, d_model, num_heads, num_layers, d_ff, num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check for existing checkpoint
    checkpoint_file = 'transformer_checkpoint.pth'
    start_epoch = 0
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    if os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        start_epoch, train_losses, train_accuracies, val_losses, val_accuracies = load_checkpoint(model, optimizer,
                                                                                                  checkpoint_file)
        start_epoch += 1  # Start from the next epoch

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, train_losses, train_accuracies, val_losses, val_accuracies,
                        checkpoint_file)

        # Plot and save training history
        plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)

    # Final evaluation on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the final model
    torch.save(model.state_dict(), "final_transformer_model.pth")


if __name__ == "__main__":
    main()