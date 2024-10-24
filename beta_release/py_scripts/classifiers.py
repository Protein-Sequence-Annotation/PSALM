import torch
from torch import nn

###########################################################
# Choices for classification head
###########################################################
    
class ClanLSTMbatch(nn.Module):

    def __init__(self, embed_dim, output_dim):
        super().__init__()

        self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=True, batch_first=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, output_dim)
        )

    def forward(self, x, mask):

        x, _ = self.lstm(x)
        x = x * mask.unsqueeze(2)
        x = self.linear_stack(x)

        return x

    def embed(self, x):

        x, _ = self.lstm(x)

        return x

class ClanLSTMbatch_onehot(nn.Module):

    def __init__(self, embed_dim, output_dim, input_dim=33):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, embed_dim, bidirectional=True, batch_first=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, output_dim)
        )

    def forward(self, x, mask):

        x, _ = self.lstm(x)
        x = x * mask.unsqueeze(2)
        x = self.linear_stack(x)

        return x

    def embed(self, x):

        x, _ = self.lstm(x)

        return x

class ClanLSTMbatch_onehot_scaled(nn.Module):

    def __init__(self, embed_dim, output_dim, input_dim=33):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, bidirectional=True, batch_first=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2*input_dim, 4*input_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*input_dim),
            nn.Linear(4*input_dim, 4*input_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*input_dim),
            nn.Linear(4*input_dim, output_dim)
        )

    def forward(self, x, mask):

        x, _ = self.lstm(x)
        x = x * mask.unsqueeze(2)
        x = self.linear_stack(x)

        return x

    def embed(self, x):

        x, _ = self.lstm(x)

        return x

class FamLSTMbatch(nn.Module): 
    def __init__(self, embed_dim, maps, device):
        super().__init__()
        self.clan_fam_matrix = maps['clan_family_matrix'].to(device)
        self.f = self.clan_fam_matrix.shape[1]

        self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, self.f)
        )

    def forward(self, x, mask, x_c):
        # Generate clan to fam weights
        weights = torch.matmul(x_c, self.clan_fam_matrix)

        # Get fam logits
        x, _ = self.lstm(x)
        x = x*mask.unsqueeze(2)
        x = self.linear_stack(x)

        # elementwise multiplication of weights and fam_vectors
        weighted_fam_vectors = x * weights

        return weighted_fam_vectors, x
    
    def embed(self, x):

        x, _ = self.lstm(x)

        return x
    
class FamLSTMbatch_onehot(nn.Module): 
    def __init__(self, embed_dim, maps, device, input_dim=33):
        super().__init__()
        self.clan_fam_matrix = maps['clan_family_matrix'].to(device)
        self.f = self.clan_fam_matrix.shape[1]

        self.lstm = nn.LSTM(input_dim, embed_dim, bidirectional=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, self.f)
        )

    def forward(self, x, mask, x_c):
        # Generate clan to fam weights
        weights = torch.matmul(x_c, self.clan_fam_matrix)

        # Get fam logits
        x, _ = self.lstm(x)
        x = x*mask.unsqueeze(2)
        x = self.linear_stack(x)

        # elementwise multiplication of weights and fam_vectors
        weighted_fam_vectors = x * weights

        return weighted_fam_vectors, x
    
    def embed(self, x):

        x, _ = self.lstm(x)

        return x

class FamLSTMbatch_onehot_scaled(nn.Module): 
    def __init__(self, embed_dim, maps, device, input_dim=33):
        super().__init__()
        self.clan_fam_matrix = maps['clan_family_matrix'].to(device)
        self.f = self.clan_fam_matrix.shape[1]

        self.lstm = nn.LSTM(input_dim, input_dim, bidirectional=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2*input_dim, 4*input_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*input_dim),
            nn.Linear(4*input_dim, 4*input_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*input_dim),
            nn.Linear(4*input_dim, self.f)
        )

    def forward(self, x, mask, x_c):
        # Generate clan to fam weights
        weights = torch.matmul(x_c, self.clan_fam_matrix)

        # Get fam logits
        x, _ = self.lstm(x)
        x = x*mask.unsqueeze(2)
        x = self.linear_stack(x)

        # elementwise multiplication of weights and fam_vectors
        weighted_fam_vectors = x * weights

        return weighted_fam_vectors, x
    
    def embed(self, x):

        x, _ = self.lstm(x)

        return x
    
class OnlyFams(nn.Module): 
    def __init__(self, embed_dim, maps, device):
        super().__init__()
        self.clan_fam_matrix = maps['clan_family_matrix'].to(device)
        self.f = self.clan_fam_matrix.shape[1]

        self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, self.f)
        )

    def forward(self, x, mask):

        # Get fam logits
        x, _ = self.lstm(x)
        x = x*mask.unsqueeze(2)
        x = self.linear_stack(x)

        return x
    
class OnlyFams_onehot(nn.Module): 
    def __init__(self, embed_dim, maps, device, input_dim=33):
        super().__init__()
        self.clan_fam_matrix = maps['clan_family_matrix'].to(device)
        self.f = self.clan_fam_matrix.shape[1]

        self.lstm = nn.LSTM(input_dim, embed_dim, bidirectional=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, self.f)
        )

    def forward(self, x, mask):

        # Get fam logits
        x, _ = self.lstm(x)
        x = x*mask.unsqueeze(2)
        x = self.linear_stack(x)

        return x