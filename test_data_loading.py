from utils import load_dataset, get_data_loaders

# Test loading datasets
print("Loading training dataset...")
train_data = load_dataset(is_training=True)
print(f"Training dataset shape: {train_data.shape}")
print(f"Training dataset columns: {train_data.columns}")

print("\nLoading test dataset...")
test_data = load_dataset(is_training=False)
print(f"Test dataset shape: {test_data.shape}")
print(f"Test dataset columns: {test_data.columns}")

# Test DataLoaders
print("\nCreating DataLoaders...")
train_loader, test_loader = get_data_loaders(batch_size=32)

print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")

# Check a single batch
for batch_features, batch_labels in train_loader:
    print(f"\nSample batch shape: {batch_features.shape}")
    print(f"Sample batch labels shape: {batch_labels.shape}")
    break

print("\nData loading test completed successfully!")