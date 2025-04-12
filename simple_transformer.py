import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class SimpleTransformer(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Configuration
        custom_config = model_config["custom_model_config"]
        self.input_dim = 76
        self.seq_len = custom_config["seq_len"]
        self.embed_size = custom_config["embed_size"]
        self.nheads = custom_config["nhead"]
        self.nlayers = custom_config["nlayers"]
        self.dropout = custom_config["dropout"]
        self.values_out = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Input layer
        self.input_embed = nn.Linear(self.input_dim, self.embed_size).to(self.device)
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(self.seq_len, self.embed_size).to(self.device)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_size,
                nhead=self.nheads,
                dropout=self.dropout,
                activation='gelu',
                device=self.device), 
            num_layers=self.nlayers
        )
        
        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(self.embed_size + 2, 64), # Add dynamic features (wallet balance, unrealized PnL)
            nn.ReLU(),
            nn.Linear(64, num_outputs) # Action space size
        ).to(self.device)

        self.value_head = nn.Sequential(
            nn.Linear(self.embed_size + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

    def forward(self, input_dict, state, seq_len):
        # Process input
        x = input_dict["obs"].view(-1, self.seq_len, self.input_dim).to(self.device)
        dynamic_features = x[:, -1, 2:4].clone().to(self.device)
        x = self.input_embed(x)
        
        position = torch.arange(0, self.seq_len).unsqueeze(0).expand(x.size(0), -1).to(self.device)
        x = x + self.pos_encoding(position)
        
        transformer_out = self.transformer(x)
        last_out = transformer_out[:, -1, :]
         
        combined = torch.cat((last_out, dynamic_features), dim=1)
        
        actions = self.policy_head(combined)
        self.values_out = self.value_head(combined).squeeze(1)
        
        return actions, state
    
    def value_function(self):
        return self.values_out
    
    