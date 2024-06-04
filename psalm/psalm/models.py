from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import safetensors
import torch
from torch import nn
import torch.nn.functional as F
import pickle
import json

class psalm_clan(nn.Module, PyTorchModelHubMixin):
    """
    A PyTorch module for the PSALM Clan classifier model.

    Args:
        input_dim (int): The input dimension of PSALM-clan.
        output_dim (int): The output dimension of PSALM-clan.

    Attributes:
        lstm (nn.LSTM): The BiLSTM layer.
        linear_stack (nn.Sequential): The sequential stack of linear layers.

    Methods:
        forward(x): Performs forward pass through the model.
        embed(x): Embeds the input sequence using the LSTM layer.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, bidirectional=True)

        self.linear_stack = nn.Sequential(
            nn.Linear(2 * input_dim, 4 * input_dim),  # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4 * input_dim),
            nn.Linear(4 * input_dim, 4 * input_dim),  # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4 * input_dim),
            nn.Linear(4 * input_dim, output_dim)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear_stack(x)
        # Get clan predictions
        x = F.softmax(x, dim=1)
        return x

    def embed(self, x):
        x, _ = self.lstm(x)
        return x

class psalm_fam(nn.Module, PyTorchModelHubMixin):
    """
    A PyTorch module for the PSALM Family classifier model.

    Args:
        input_dim (int): The input dimension of the PSALM-family.
        output_dim (int): The output dimension of the PSALM_family.

    Attributes:
        lstm (nn.LSTM): The bidirectional BiLSTM layer.
        linear_stack (nn.Sequential): The sequential stack of linear layers.

    Methods:
        forward(x, x_clan, clan_family_matrix): Performs forward pass of the model conditioned on clan predictions/annotations.
        from_pretrained(model_name): Loads a pretrained model.

    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, bidirectional=True)
        self.linear_stack = nn.Sequential(
            nn.Linear(2*input_dim, 4*input_dim),  # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*input_dim),
            nn.Linear(4*input_dim, 4*input_dim),  # Everything x2 because biLSTM
            nn.ReLU(),
            nn.LayerNorm(4*input_dim),
            nn.Linear(4*input_dim, output_dim)
        )

    def forward(self, x, clan_preds, clan_family_matrix):
        """
        Performs forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            x_clan (torch.Tensor): The clan tensor.
            clan_family_matrix (torch.Tensor): The clan-family matrix.

        Returns:
            torch.Tensor: The predicted family logits.

        """
        # Get fam logits
        x, _ = self.lstm(x)
        fam_preds = self.linear_stack(x)

        # Apply softmax to the family predictions based on the clan_family_matrix
        for i in range(clan_family_matrix.shape[0]):  # shape is cxf
            indices = torch.nonzero(clan_family_matrix[i]).squeeze()  # indices for the softmax
            if i == clan_family_matrix.shape[0]-1:
                fam_preds[:, indices] = 1  # IDR is 1:1 map
            else:
                fam_preds[:, indices] = torch.softmax(fam_preds[:, indices], dim=1)

        # Multiply by clan, expand clan preds
        clan_preds_f = torch.matmul(clan_preds, clan_family_matrix)

        # Element wise mul of preds and clan
        fam_preds = fam_preds * clan_preds_f

        return fam_preds

    @classmethod
    def from_pretrained(cls, model_name):
        """
        Loads a pretrained model.

        Args:
            model_name (str): The name of the pretrained model.

        Returns:
            Tuple[psalm_fam, Dict]: A tuple containing the loaded model and the maps dictionary.

        """
        # Download the model's weights
        model_file = hf_hub_download(repo_id=f"{model_name}", filename="model.safetensors")
        # Read the content of the file as bytes
        with open(model_file, 'rb') as f:
            model_data = f.read()
        # Load the model's weights
        state_dict = safetensors.torch.load(model_data)

        # Load the model's configuration
        config_file = hf_hub_download(repo_id=f"{model_name}", filename="config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Download the maps dictionary
        maps_file = hf_hub_download(repo_id=f"{model_name}", filename="maps.pkl")
        with open(maps_file, 'rb') as f:
            maps = pickle.load(f)

        # Create a new instance of the model class using the configuration and load the weights and maps
        model = cls(config["input_dim"], config["output_dim"])
        model.load_state_dict(state_dict)

        return model, maps