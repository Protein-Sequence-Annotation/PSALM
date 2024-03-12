import torch
from torch import nn
import torch.nn.functional as F

###########################################################
# Choices for classification head
###########################################################

class LinearHead3Normed(nn.Module):
    """
    Linear classification head with one layer and Layer Normalization
    """

    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                  nn.LayerNorm(hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, hid_dim),
                                  nn.LayerNorm(hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, out_dim))

    def forward(self, x):
        return self.head(x)

class LinearHead3NormedFamSoft(nn.Module):
    """
    Linear classification head with one layer and Layer Normalization
    """

    def __init__(self, embed_dim, clan_dim, fam_dim):
        super().__init__()
        self.clan_predictor = nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),
                                  nn.LayerNorm(2*embed_dim),
                                  nn.ReLU(),
                                  nn.Linear(2*embed_dim, 2*embed_dim),
                                  nn.LayerNorm(2*embed_dim),
                                  nn.ReLU(),
                                  nn.Linear(2*embed_dim, clan_dim))

        self.fam_predictor = nn.Sequential(nn.Linear(embed_dim + clan_dim, 2*(embed_dim + clan_dim)),
                                   nn.LayerNorm(2*(embed_dim + clan_dim)),
                                   nn.ReLU(),
                                   nn.Linear(2*(embed_dim + clan_dim), 2*(embed_dim + clan_dim)),
                                   nn.LayerNorm(2*(embed_dim + clan_dim)),
                                   nn.ReLU(),
                                   nn.Linear(2*(embed_dim + clan_dim), fam_dim))

    def forward(self, x):
        # x is of shape (L, d)

        # Predict clan_vector
        clan_vector = F.softmax(self.clan_predictor(x), dim=1)  # shape: (L, c)

        # Concatenate x and clan_vector
        x_concat = torch.cat((x, clan_vector), dim=1)  # shape: (L, d+c)

        # Predict fam_vector
        fam_vector = self.fam_predictor(x_concat)  # shape: (L, f)

        return fam_vector, clan_vector
    
class FamModelMoE(nn.Module): 
    def __init__(self, embed_dim, maps, device):
        super().__init__()
        self.d = embed_dim
        self.maps = maps
        self.c = len(self.maps['clan_idx'].keys()) # number of clans
        self.f = len(self.maps['fam_idx'].keys()) # number of families
        self.f_list = [0] * self.c
        for clan, idx in self.maps["clan_idx"].items():
            self.f_list[idx] = len(self.maps["clan_fam"][clan]) # list of number of families in each clan in the order of clan_idx
        self.clan_fam_matrix = maps['clan_family_matrix'].to(device)
         
        # For each clan, we create a linear layer that takes an input of size d and produces an output of size f_i (the number of families in the clan).
        self.fam_predictors = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, 2*f_i),
            nn.LayerNorm(2*f_i),
            nn.ReLU(),
            nn.Linear(2*f_i, f_i)
        ) for f_i in self.f_list]) # Last predictor is for IDR 

    def forward(self, x,x_c):
        # Assuming f_list is a list of indices that specifies the order in which to apply the fam_predictors
        # and x_c is a tensor of shape (L, c) that contains the weights for each fam_predictor

        max_clan_indices = torch.argmax(x_c, dim=1)
        output = torch.zeros(x.shape[0], self.f).to(x.device) #Lxf

        unique_clans = torch.unique(max_clan_indices)

        for clan_idx in unique_clans:
            mask = torch.where(max_clan_indices == clan_idx)[0]

            expert = self.fam_predictors[clan_idx]
            clan_id = self.maps['idx_clan'][int(clan_idx)]
            fam_ids = self.maps['clan_fam'][clan_id]
            fam_idxs = [self.maps['fam_idx'][fam_id] for fam_id in fam_ids]

            output[mask[:,None], fam_idxs] = expert(x[mask]) #advanced indexing
            break

        return output
    
class FamModelMoELSTM(nn.Module): 
    def __init__(self, embed_dim, maps, device):
        super().__init__()
        self.d = embed_dim
        self.maps = maps
        self.c = len(self.maps['clan_idx'].keys()) # number of clans
        self.f = len(self.maps['fam_idx'].keys()) # number of families
        self.f_list = [0] * self.c
        for clan, idx in self.maps["clan_idx"].items():
            self.f_list[idx] = len(self.maps["clan_fam"][clan]) # list of number of families in each clan in the order of clan_idx
        self.clan_fam_matrix = maps['clan_family_matrix'].to(device)
        self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=True)
        # For each clan, we create a linear layer that takes an input of size d and produces an output of size f_i (the number of families in the clan).
        self.fam_predictors = nn.ModuleList([nn.Sequential(
            nn.Linear(2*embed_dim, 2*f_i),
            nn.LayerNorm(2*f_i),
            nn.ReLU(),
            nn.Linear(2*f_i, f_i)
        ) for f_i in self.f_list]) # Last predictor is for IDR 

    def forward(self, x,x_c):
        # Assuming f_list is a list of indices that specifies the order in which to apply the fam_predictors
        # and x_c is a tensor of shape (L, c) that contains the weights for each fam_predictor
        x,_ = self.lstm(x)
        max_clan_indices = torch.argmax(x_c, dim=1)
        output = torch.zeros(x.shape[0], self.f).to(x.device) #Lxf

        unique_clans = torch.unique(max_clan_indices)

        for clan_idx in unique_clans:
            mask = torch.where(max_clan_indices == clan_idx)[0]

            expert = self.fam_predictors[clan_idx]
            clan_id = self.maps['idx_clan'][int(clan_idx)]
            fam_ids = self.maps['clan_fam'][clan_id]
            fam_idxs = [self.maps['fam_idx'][fam_id] for fam_id in fam_ids]

            output[mask[:,None], fam_idxs] = expert(x[mask]) #advanced indexing
            break

        return output

class FamModelSimple(nn.Module): 
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

    def forward(self, x,x_c):
        # Generate clan to fam weights
        weights = torch.matmul(x_c, self.clan_fam_matrix)

        # Get fam logits
        x, _ = self.lstm(x)
        x = self.linear_stack(x)

        # elementwise multiplication of weights and fam_vectors
        weighted_fam_vectors = x * weights

        return weighted_fam_vectors, x

class ClanLSTM(nn.Module):

    def __init__(self, embed_dim, output_dim):
        super().__init__()

        self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, output_dim)
        )

    def forward(self, x):

        x, _ = self.lstm(x)
        x = self.linear_stack(x)

        return x
    
class ClanFamLSTM(nn.Module):

    def __init__(self, embed_dim, output_dim, maps, device): # output_dim is clan count
        super().__init__()

        self.clan_fam_matrix = maps['clan_family_matrix'].to(device)
        self.f = self.clan_fam_matrix.shape[1]

        self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=True)

        self.clan_linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, output_dim)
        )

        self.fam_linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, 4*embed_dim), # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),
            nn.Linear(4*embed_dim, self.f)
        )

    def forward(self, x):

        x, _ = self.lstm(x) # Run through common LSTM
        x_c = self.clan_linear_stack(x) # Get clan prediction

        weights = torch.matmul(x_c, self.clan_fam_matrix) # Decide focus positions

        x_f = self.fam_linear_stack(x) # Get family prediction

        # elementwise multiplication of weights and fam_vectors
        weighted_fam_vectors = x_f * weights

        return x_c, weighted_fam_vectors, x_f