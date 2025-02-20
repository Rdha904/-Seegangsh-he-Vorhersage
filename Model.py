import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_output, encoder_outputs):
        # Compute alignment scores
        alignment_scores = torch.bmm(encoder_outputs, decoder_output.transpose(1, 2))
        
        # Compute attention weights (softmax over alignment scores)
        attn_weights = F.softmax(alignment_scores, dim=1)
        
        # Compute context vector as weighted sum of encoder outputs
        context_vector = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        
        return context_vector, attn_weights

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.attention = LuongAttention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Concatenate context vector with LSTM output

    def forward(self, x, hidden, cell, encoder_outputs):
        # LSTM output
        lstm_output, (hidden, cell) = self.lstm(x, (hidden, cell))
        
        # Calculate attention and context vector
        context_vector, _ = self.attention(lstm_output, encoder_outputs)
        
        # Concatenate LSTM output with context vector
        combined = torch.cat((lstm_output, context_vector), dim=2)
        
        # Apply TimeDistributed Dense Layer (fc) manually for each time step
        batch_size, seq_len, _ = combined.size()
        combined = combined.view(batch_size * seq_len, -1)
        output = self.fc(combined)
        output = output.view(batch_size, seq_len, -1)
        
        return output, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, n_layers, dropout)
        self.decoder = Decoder(hidden_size, output_size, n_layers, dropout)

    def forward(self, src, trg):
        # Encode the input sequence
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Use the last hidden state of the encoder as the initial input for the decoder
        input = hidden[-1].unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Repeat the hidden state across the desired sequence length
        input = input.repeat(1, trg.size(1), 1)  # (batch_size, trg_len, hidden_size)
        
        # Decode the sequence
        output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
        
        return output