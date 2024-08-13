import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import TrojanDataset, get_data_loaders
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime

print("Script is running")

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # * 2 for bidirectional
        self.dropout = nn.Dropout(dropout_rate)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.size()

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # * 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None, checkpoint_path='checkpoint.pth', results_path='results.json'):
    print("Starting training...")
    train_losses = []
    test_accuracies = []
    best_accuracy = 0
    start_epoch = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        test_accuracies = checkpoint['test_accuracies']
        best_accuracy = checkpoint['best_accuracy']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Duration: {epoch_duration:.2f} seconds')

        # Adjust learning rate
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'best_accuracy': best_accuracy
        }, checkpoint_path)

        # Save results
        results = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'best_accuracy': best_accuracy
        }
        with open(results_path, 'w') as f:
            json.dump(results, f)

        # Update best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training completed.")
    return train_losses, test_accuracies

def plot_results(train_losses, test_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def main():
    print("Starting main function...")
    # Hyperparameters
    input_size = 90
    hidden_size = 256
    num_layers = 3
    num_classes = 2
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    dropout_rate = 0.5
    print("Hyperparameters set.")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data loaders
    print("Loading data...")
    train_loader, test_loader = get_data_loaders(batch_size)
    print(f"Data loaded. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Initialize the model
    print("Initializing model...")
    model = ImprovedLSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_rate).to(device)
    print("Model initialized.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    print("Loss, optimizer, and scheduler set up.")

    # Train the model
    print("Starting model training...")
    start_time = time.time()
    train_losses, test_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs,
                                                device, scheduler)
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Model training completed. Total training time: {training_duration:.2f} seconds")

    # Plot results
    print("Plotting results...")
    plot_results(train_losses, test_accuracies)
    print("Results plotted and saved.")

    # Final model save
    print("Saving final model...")
    torch.save(model.state_dict(), 'final_lstm_model.pth')
    print("Final model saved.")

    print("Main function completed.")

if __name__ == '__main__':
    print(f"Starting script at {datetime.now()}")
    main()
    print(f"Script completed at {datetime.now()}")