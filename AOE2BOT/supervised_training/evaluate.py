import torch
from train import model, val_loader, criterion  # Assuming model is trained

def evaluate():
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
    accuracy = correct / total
    print(f'Evaluation - Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}')
    return accuracy

if __name__ == '__main__':
    evaluate()