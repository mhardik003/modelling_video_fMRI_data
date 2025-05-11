# src/models.py
import torch
import torch.nn as nn

# class EncodingModel(nn.Module):
#     """Predicts fMRI activity from video embeddings."""
#     def __init__(self, video_embed_dim: int, fmri_dim: int, hidden_dim: int = 1024):
#         super().__init__()
#         self.fc1 = nn.Linear(video_embed_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, fmri_dim)

#     def forward(self, video_embeddings):
#         """
#         Args:
#             video_embeddings (Tensor): Shape (batch_size, time_steps, video_embed_dim)
#                                         or (batch_size, video_embed_dim) if time distributed applied outside.
#                                         Assume (batch_size, video_embed_dim) for now.
#         Returns:
#             Tensor: Predicted fMRI data. Shape (batch_size, fmri_dim)
#         """
#         x = self.fc1(video_embeddings)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x



class VoxelWiseEncodingModel(nn.Module):
    """
    Voxel-wise regression model: one small MLP per voxel.
    """
    def __init__(self, video_embed_dim: int, fmri_dim: int, hidden_dim: int = 128, depth: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.voxel_models = nn.ModuleList([
            self._make_voxel_mlp(video_embed_dim, hidden_dim, depth, dropout_rate)
            for _ in range(fmri_dim)
        ])

    def _make_voxel_mlp(self, input_dim, hidden_dim, depth, dropout_rate):
        layers = []
        for i in range(depth):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, video_embeddings):
        """
        video_embeddings: Tensor of shape (num_timestamps, video_embed_dim)
        Returns:
            fMRI predictions: Tensor of shape (num_timestamps, fmri_dim)
        """
        voxel_outputs = [
            voxel_model(video_embeddings).squeeze(-1)
            for voxel_model in self.voxel_models
        ]
        return torch.stack(voxel_outputs, dim=-1)




#<-------------------------------------------------- MLP ------------------------------------------------->

class EncodingModel(nn.Module):
    """Deep neural network to predict fMRI activity from video embeddings."""
    def __init__(self, video_embed_dim: int, fmri_dim: int, hidden_dim: int = 1024, depth: int = 8, dropout_rate: float = 0.3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(video_embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),


            nn.Linear(512, fmri_dim)
        )

    def forward(self, video_embeddings):
        
        return self.model(video_embeddings)
    
#<---------------------------------------------------------------------------------------------------------->

#<----------------------------------------------- LSTM ----------------------------------------------------->
class LSTMEncodingModel(nn.Module):

    def __init__(self, video_embed_dim: int, fmri_dim: int, hidden_dim: int = 1024, 
                 num_layers: int = 2, bidirectional: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        
        # self.input_layers = nn.Sequential(
        #     nn.Linear(video_embed_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate)
        # )
        # Initial dimensionality reduction for video data

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=video_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Calculate the output dimension from LSTM
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Final prediction layers
        self.output_layers = nn.Sequential(
            nn.BatchNorm1d(lstm_out_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, fmri_dim)
        )

    def forward(self, video_embeddings):
        """
        Args:
            video_embeddings (Tensor): Shape (batch_size, seq_length, video_embed_dim)
                                       If your input is just (batch_size, video_embed_dim),
                                       unsqueeze to add sequence dimension.
        Returns:
            Tensor: Predicted fMRI data. Shape (batch_size, fmri_dim)
        """
        # Check if we need to add a sequence dimension
        if len(video_embeddings.shape) == 2:
            video_embeddings = video_embeddings.unsqueeze(1)  # Add seq_length dimension
            
        # Process through LSTM
        lstm_out, _ = self.lstm(video_embeddings)
        
        # Take the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Final prediction
        fmri_prediction = self.output_layers(last_output)
        
        return fmri_prediction
    
#<------------------------------------------------------------------------------------------------------------------->

# class DecodingModel(nn.Module):
#     """Predicts video embeddings from fMRI activity."""
#     def __init__(self, fmri_dim: int, video_embed_dim: int, hidden_dim: int = 2048):
#         super().__init__()
#         self.fc1 = nn.Linear(fmri_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, video_embed_dim)

#     def forward(self, fmri_data):
#         """
#         Args:
#             fmri_data (Tensor): Shape (batch_size, time_steps, fmri_dim)
#                                 or (batch_size, fmri_dim). Assume (batch_size, fmri_dim).
#         Returns:
#             Tensor: Predicted video embeddings. Shape (batch_size, video_embed_dim)
#         """
#         x = self.fc1(fmri_data)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

#<------------------------------------------------- MLP --------------------------------------------->

class DecodingModel(nn.Module):
    """Deep neural network to predict video embeddings from fMRI activity."""
    def __init__(self, fmri_dim: int, video_embed_dim: int, hidden_dim: int = 2048, depth: int = 8, dropout_rate: float = 0.4):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(fmri_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, video_embed_dim)
        )


    def forward(self, fmri_data):
        
        return self.model(fmri_data)

#<------------------------------------------------------------------------------------------------------------------->

#<------------------------------------------------ LSTM ------------------------------------------------------------->

class LSTMDecodingModel(nn.Module):
    """Deep neural network to predict video embeddings from fMRI activity."""
    def __init__(self, fmri_dim: int, video_embed_dim: int, hidden_dim: int = 1024, 
                 num_layers: int = 2, dropout_rate: float = 0.2, bidirectional: bool = True):
        super().__init__()
        
        # Initial dimensionality reduction for fMRI data
        self.input_proj = nn.Sequential(
            nn.Linear(fmri_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, video_embed_dim)
        )
        
    def forward(self, fmri_data):
        
        is_single_timepoint = fmri_data.dim() == 2
        
        if is_single_timepoint:
            # Add sequence dimension if single timepoint
            fmri_data = fmri_data.unsqueeze(1)  # (batch_size, 1, fmri_dim)
        
        batch_size, seq_length, _ = fmri_data.shape
        
        fmri_flat = fmri_data.reshape(-1, fmri_data.size(-1))  # (batch_size*seq_length, fmri_dim)
        
        projected = self.input_proj(fmri_flat)
        
        projected = projected.reshape(batch_size, seq_length, -1)  # (batch_size, seq_length, hidden_dim)
        
        lstm_out, _ = self.lstm(projected)  # (batch_size, seq_length, hidden_dim*2 if bidirectional)
        
        lstm_out_flat = lstm_out.reshape(-1, lstm_out.size(-1))  # (batch_size*seq_length, hidden_dim*2)
        
        out = self.output_proj(lstm_out_flat)  # (batch_size*seq_length, video_embed_dim)
        
        out = out.reshape(batch_size, seq_length, -1)  # (batch_size, seq_length, video_embed_dim)
        
        if is_single_timepoint:
            out = out.squeeze(1)  # (batch_size, video_embed_dim)
        
        return out

#<-------------------------------------------------------------------------------------------------------------------->

# class SubjectAwareLSTMEncodingModel(nn.Module):
#     """LSTM network to predict fMRI activity from video embeddings with subject separation."""
#     def __init__(self, video_embed_dim: int, fmri_dim: int, hidden_dim: int = 1024, 
#                  num_layers: int = 2, bidirectional: bool = True, dropout_rate: float = 0.2):
#         super().__init__()
        
#         # LSTM layers
#         self.lstm = nn.LSTM(
#             input_size=video_embed_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional,
#             dropout=dropout_rate if num_layers > 1 else 0
#         )
        
#         # Calculate the output dimension from LSTM
#         lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
#         # Final prediction layers
#         self.output_layers = nn.Sequential(
#             nn.BatchNorm1d(lstm_out_dim),
#             nn.Dropout(dropout_rate),
#             nn.Linear(lstm_out_dim, hidden_dim),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, fmri_dim)
#         )

#     def forward(self, video_embeddings, subject_boundaries=None):
#         """
#         Forward pass with awareness of subject boundaries.
        
#         Args:
#             video_embeddings (Tensor): Shape (batch_size, seq_length, video_embed_dim)
#             subject_boundaries (List[int], optional): Indices where subject data changes.
#                                                     E.g., [100, 250] means first subject has
#                                                     indices 0-99, second 100-249, etc.
        
#         Returns:
#             Tensor: Predicted fMRI data. Shape (batch_size, fmri_dim)
#         """
#         batch_size, seq_length, embed_dim = video_embeddings.shape
        
#         if subject_boundaries is None or len(subject_boundaries) == 0:
#             # No subject boundaries provided, process as a single sequence
#             lstm_out, _ = self.lstm(video_embeddings)
#             last_outputs = lstm_out[:, -1, :]  # Take the last output for each sequence
#         else:
#             # Process each subject's data separately to avoid context leakage
#             device = video_embeddings.device
#             all_outputs = []
            
#             # Add start and end indices for convenience
#             boundaries = [0] + subject_boundaries + [batch_size]
            
#             for i in range(len(boundaries) - 1):
#                 start_idx = boundaries[i]
#                 end_idx = boundaries[i + 1]
                
#                 # Skip if no samples for this subject
#                 if end_idx <= start_idx:
#                     continue
                
#                 # Process this subject's data
#                 subject_data = video_embeddings[start_idx:end_idx]
                
#                 # Reset hidden state for each subject
#                 lstm_out, _ = self.lstm(subject_data)
                
#                 # Take the last output for each sequence in this subject
#                 subject_outputs = lstm_out[:, -1, :]
#                 all_outputs.append(subject_outputs)
            
#             # Concatenate all subject outputs
#             last_outputs = torch.cat(all_outputs, dim=0)
        
#         # Final prediction
#         fmri_prediction = self.output_layers(last_outputs)
        
#         return fmri_prediction