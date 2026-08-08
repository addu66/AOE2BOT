import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from preprocess import preprocess_all
from data_loader import load_all_replays
from model import ActionPredictor

class ReplayDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    states = torch.tensor([item['state'] for item in batch], dtype=torch.float)
    seqs = [torch.tensor(item['seq'], dtype=torch.long) for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return states, seqs_padded, labels

# Load data
replays = load_all_replays()
data, action_vocab = preprocess_all(replays)
vocab_size = len(action_vocab)

# Split data (simple 80/20)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

train_dataset = ReplayDataset(train_data)
val_dataset = ReplayDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Model
model = ActionPredictor(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for states, seqs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(states, seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for states, seqs, labels in val_loader:
            outputs = model(states, seqs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Val Loss: {val_loss / len(val_loader)}, Accuracy: {correct / total}')