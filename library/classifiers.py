import torch
from torch import nn
import torch.nn.functional as F

###########################################################
# Choices for classification head
###########################################################

class LinearHead1(nn.Module):

    """
    Linear classification head with one layer
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):

        return self.head(x)

class LinearHead3(nn.Module):

    """
    Linear classification head with one layer
    """

    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()

        self.head = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, out_dim))

    def forward(self, x):

        return self.head(x)
    
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

class LinearHead4Normed(nn.Module):
    """
    Linear classification head with four layers and Layer Normalization
    """

    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                  nn.LayerNorm(hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, hid_dim),
                                  nn.LayerNorm(hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, hid_dim),
                                  nn.LayerNorm(hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, out_dim))

    def forward(self, x):
        return self.head(x)

class LinearHead3NormedFam(nn.Module):
    """
    Linear classification head with one layer and Layer Normalization
    """

    def __init__(self, embed_dim, clan_dim, fam_dim, n_heads, attend=True):
        super().__init__()
        self.attend = attend
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
        if self.attend:
            self.attention = nn.MultiheadAttention(embed_dim + clan_dim, n_heads)

    def forward(self, x):
        # x is of shape (L, d)

        # Predict clan_vector
        clan_vector = self.clan_predictor(x)  # shape: (L, c)

        # Concatenate x and clan_vector
        x_concat = torch.cat((x, clan_vector), dim=1)  # shape: (L, d+c)

        # Apply attention
        if self.attend:
            attn_output, _ = self.attention(x_concat.unsqueeze(0), x_concat.unsqueeze(0), x_concat.unsqueeze(0))
            x_concat = attn_output.squeeze(0)  # shape: (L, d+c)

        # Predict fam_vector
        fam_vector = self.fam_predictor(x_concat)  # shape: (L, f)

        return fam_vector, clan_vector

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


class LinearHead3Normed_Seq(nn.Module):
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
        x = x.mean(dim=0)
        return self.head(x)

class ContextConcatLinear3(nn.Module):
    def __init__(self, d, c):
        super(ContextConcatLinear3, self).__init__()
        self.d = d
        self.c = c
        self.linear_stack = nn.Sequential(
            nn.Linear(3*d, 4*d),
            nn.ReLU(),
            nn.Linear(4*d, 2*d),
            nn.ReLU(),
            nn.Linear(2*d, c)
        )

    def forward(self, x):
        # Compute full_pooled_embed
        full_pooled_embed = torch.mean(x, dim=0)


        # Compute local_pooled_embed
        # Reshape the input tensor to have the shape (1, d, L)
        x_reshaped = x.transpose(0, 1).unsqueeze(0)

        # Apply 1D average pooling
        local_pooled_embed = torch.nn.functional.avg_pool1d(x_reshaped, kernel_size=21, stride=1, padding=10, count_include_pad=False)

        #Remove the extra batch dimension and transpose the tensor back to shape (L, d)
        local_pooled_embed = local_pooled_embed.squeeze(0).transpose(0, 1)

        # Concatenate embeddings
        concat_embed = torch.cat([full_pooled_embed.repeat(x.shape[0], 1), local_pooled_embed, x], dim=1)

        # Pass through linear layers
        output = self.linear_stack(concat_embed)

        return output

class ContextWeightedSum(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()

        # Define the weights as nn.Parameter. Each weight is a 1D tensor of size embed_dim
        self.weights = nn.Parameter(torch.ones(4, 1, embed_dim))

        # Define the linear stack with layer norms
        self.linear_stack = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.LayerNorm(2*embed_dim),
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.LayerNorm(2*embed_dim),
            nn.Linear(2*embed_dim, output_dim)
        )

    def forward(self, x):
        # Compute the full pooled embed
        full_pooled_embed = torch.mean(x, dim=0)
        # Expand full_pooled_embed to match the shape of the other tensors
        full_pooled_embed = full_pooled_embed.expand(x.size(0), -1)

        # Compute the local pooled embeds
        # Reshape the input tensor to have the shape (1, d, L)
        x_reshaped = x.transpose(0, 1).unsqueeze(0)
        local_pooled_embed_2 = F.avg_pool1d(x_reshaped, kernel_size=5, stride=1, padding=2, count_include_pad=False)
        local_pooled_embed_4 = F.avg_pool1d(x_reshaped, kernel_size=9, stride=1, padding=4, count_include_pad=False)

        #Remove the extra batch dimension and transpose the tensor back to shape (L, d)
        local_pooled_embed_2 = local_pooled_embed_2.squeeze(0).transpose(0, 1)
        local_pooled_embed_4 = local_pooled_embed_4.squeeze(0).transpose(0, 1)

        # Stack the inputs along a new dimension
        stacked_embeds = torch.stack([full_pooled_embed, local_pooled_embed_2, local_pooled_embed_4, x], dim=0)

        # Compute a weighted average of the previous 4 quantities
        weighted_embeds = stacked_embeds * self.weights
        weighted_average = torch.sum(weighted_embeds, dim=0) / (torch.sum(self.weights) + 1e-10)  # Add a small constant to prevent division by zero

        # Pass the weighted average through the linear stack
        output = self.linear_stack(weighted_average)

        return output

class EnhancedContextWeightedSum(nn.Module):
    def __init__(self, embed_dim, output_dim, kernel_sizes=(5, 9)):
        super().__init__()
        '''
        this is something copilot wrote
        experiment with using conv1d and LSTM (uni or bi) to extract local and global features
        could add them or process sequentially or concatenate
        '''
        # Convolutional layer to extract local features
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        # LSTM layer to capture global information, now bidirectional
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        # Multi-head attention mechanism to weigh the importance of different parts of the input
        self.attention = nn.MultiheadAttention(2*embed_dim, num_heads=4)  # Adjusted for bidirectional LSTM

        # Weights for the weighted sum of the embeddings
        self.weights = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(4, 1, 2*embed_dim)))  # Adjusted for bidirectional LSTM
        # Kernel sizes for the average pooling operations
        self.kernel_sizes = kernel_sizes

        # Linear layers for the final classification
        self.linear_stack = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim),  # Adjusted for bidirectional LSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),  # Adjusted for bidirectional LSTM
            nn.Linear(4*embed_dim, 4*embed_dim),  # Adjusted for bidirectional LSTM
            nn.ReLU(),
            nn.LayerNorm(4*embed_dim),  # Adjusted for bidirectional LSTM
            nn.Linear(4*embed_dim, output_dim)  # Adjusted for bidirectional LSTM
        )

    def forward(self, x):
        # Store the original input for the skip connection
        original_x = x
        # Rearrange the dimensions of the input for the convolutional layer
        x = x.transpose(0, 1).unsqueeze(0)
        # Apply the convolutional layer
        x = self.conv(x)
        # Apply the LSTM layer
        x, _ = self.lstm(x)
        # Apply the attention mechanism
        x, _ = self.attention(x, x, x)
        # Rearrange the dimensions of the output back to the original order
        x = x.squeeze(0).transpose(0, 1)

        # Add the original input to the output of the attention mechanism (skip connection)
        x = x + original_x

        # Compute the full average pooling of the embeddings
        full_pooled_embed = torch.mean(x, dim=0)
        # Compute the local average pooling of the embeddings with the first kernel size
        local_pooled_embed_2 = F.avg_pool1d(x.transpose(0, 1).unsqueeze(0), kernel_size=self.kernel_sizes[0], stride=1, padding=self.kernel_sizes[0]//2, count_include_pad=False)
        # Compute the local average pooling of the embeddings with the second kernel size
        local_pooled_embed_4 = F.avg_pool1d(x.transpose(0, 1).unsqueeze(0), kernel_size=self.kernel_sizes[1], stride=1, padding=self.kernel_sizes[1]//2, count_include_pad=False)

        # Rearrange the dimensions of the local average pooled embeddings
        local_pooled_embed_2 = local_pooled_embed_2.squeeze(0).transpose(0, 1)
        local_pooled_embed_4 = local_pooled_embed_4.squeeze(0).transpose(0, 1)

        # Stack the full and local average pooled embeddings and the original embeddings
        stacked_embeds = torch.stack([full_pooled_embed, local_pooled_embed_2, local_pooled_embed_4, x], dim=0)

        # Compute the weighted sum of the embeddings
        weighted_embeds = stacked_embeds * self.weights
        weighted_average = torch.sum(weighted_embeds, dim=0) / (torch.sum(self.weights) + 1e-10)

        # Apply the linear layers to get the final output
        output = self.linear_stack(weighted_average)

        return output
    
class TryLSTM(nn.Module):

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