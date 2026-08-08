import torch
import torch.nn as nn

class ActionPredictor(nn.Module):
    def __init__(self, vocab_size, state_dim=100, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.state_fc = nn.Linear(state_dim, hidden_dim)  # Project state to hidden dim
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # Concat LSTM and state
    
    def forward(self, state, seq):
        # state: [batch, state_dim]
        # seq: [batch, seq_len]
        embeds = self.embedding(seq)  # [batch, seq_len, embed_dim]
        _, (h, _) = self.lstm(embeds)  # h: [1, batch, hidden_dim]
        h = h.squeeze(0)  # [batch, hidden_dim]
        state_proj = self.state_fc(state)  # [batch, hidden_dim]
        combined = torch.cat([h, state_proj], dim=1)  # [batch, hidden_dim*2]
        out = self.fc(combined)  # [batch, vocab_size]
        return out