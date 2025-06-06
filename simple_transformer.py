import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.models.modelv2 import ModelV2

@DeveloperAPI
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
        self.device = None
        self.cnn_enabled = custom_config["cnn_enabled"]
        self.freeze_cnn = custom_config["freeze_cnn"]

        # -------------- CNN Front-end --------------------
        if self.cnn_enabled:
            self.hourly_cnn = nn.Sequential(
                nn.Conv1d(36, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
                nn.GELU(),
                nn.Conv1d(64, self.embed_size, kernel_size=1)
            )
            
            # self.daily_cnn = nn.Sequential(
            #     nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1),
            #     nn.GELU(),
            #     nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            #     nn.GELU(),
            #     nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            #     nn.GELU(),
            #     nn.Conv1d(64, self.embed_size, kernel_size=1, padding=1)
            # )
            
            # # load pre-trained parameters
            # map_loc = "cuda" if torch.cuda.is_available() else "cpu"
            # hourly_state = torch.load("hourly_cnn_pretrain.pt", map_location=map_loc)
            # # daily_state = torch.load("daily_cnn_pretrain.pt")
            
            # # strip the “cnn.” prefix:
            # hourly_state = { k.replace("cnn.", ""): v for k,v in hourly_state.items() }
            # # daily_state = { k.replace("cnn.", ""): v for k,v in daily_state.items() }
            
            # self.hourly_cnn.load_state_dict(hourly_state, strict=True)
            # self.daily_cnn.load_state_dict(daily_state, strict=True)
            
            # freeze parameters
            if self.freeze_cnn:
                for p in self.hourly_cnn.parameters():
                    p.requires_grad = False
                # for p in self.daily_cnn.parameters():
                #     p.requires_grad = False
        else:
            self.cnn = None 
            
        # -------------- Input layer -----------------------
        self.input_embed = nn.Linear(self.input_dim, self.embed_size)
        # 1-1 projection (if cnn is used)
        self.projection = nn.Linear(2*self.embed_size, self.embed_size)
        # Positional encoding
        self.pos_encoding = nn.Embedding(self.seq_len, self.embed_size)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_size,
                nhead=self.nheads,
                dropout=self.dropout,
                activation='gelu',
                norm_first=True,
                batch_first=True), 
            num_layers=self.nlayers
        )
        
        # ---------- Policy and value networks ------------------
        self.policy_head = nn.Sequential(
            nn.Linear(self.embed_size + 2, 256),  # Add dynamic features (wallet balance, unrealized PnL)
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_outputs) # Action space size
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.embed_size + 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def split_hourly_daily(self, window: torch.Tensor):
        
        # Indices for hourly and daily features
        hourly_idx = [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        # daily_idx = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        # 39, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]

        # Extract hourly and daily data
        hourly_data = window[:, :, hourly_idx]
        # daily_data_full = window[:, :, daily_idx]

        # For daily data, select rows where hour == 0 (start of each day)
        # N, T, D = daily_data_full.shape
        # days = T//24

        # daily_data_reshaped = daily_data_full.reshape(N, days, 24, D)
        # daily_data = daily_data_reshaped[:, :, 0, :]

        return hourly_data #daily_data
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self.device = input_dict["obs"].device
        x = input_dict["obs"].view(-1, self.seq_len, self.input_dim).to(self.device)
        dynamic_features = x[:, -1, 2:4].clone()
    
        if self.cnn_enabled:
            hourly_data = self.split_hourly_daily(x).to(self.device)
            hourly_feat_maps = self.hourly_cnn(hourly_data.permute(0, 2, 1))  # (N, embed_size, seq_len)
    
        x = self.input_embed(x)
        if self.cnn_enabled:
            concat = torch.cat((hourly_feat_maps.permute(0, 2, 1), x), dim=2) # (N, seq_len, 2*embed_size)
            
            x = self.projection(concat) # projection back to embed_size
            
        position = torch.arange(0, self.seq_len, device=self.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pos_encoding(position)
    
        transformer_out = self.transformer(x)
        last_out = transformer_out[:, -1, :]
        combined = torch.cat((last_out, dynamic_features), dim=1)
        logits = self.policy_head(combined)
        
        self.values_out = self.value_head(combined).squeeze(1)
    
        return logits, state
    
    @override(ModelV2)
    def value_function(self):
        return self.values_out.to(self.device)
    