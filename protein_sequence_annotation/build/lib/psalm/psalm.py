import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import safetensors
import torch
from torch import nn
import esm
from Bio import SeqIO
import torch.nn.functional as F
import pickle
import json
from .viz_utils import *
import os

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
        with torch.no_grad():
            x, _ = self.lstm(x)
            x = self.linear_stack(x)
            # Get clan predictions
            x = F.softmax(x, dim=1)
        return x

    def embed(self, x):
        with torch.no_grad():
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
        with torch.no_grad():
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

class psalm:
    """
    The `psalm` class is used for protein annotation using the PSALM algorithm.

    Parameters:
    - clan_model_name (str): The name of the pretrained clan model.
    - fam_model_name (str): The name of the pretrained fam model.
    - threshold (float, optional): The threshold value for prediction confidence. Default is 0.72.
    - device (str, optional): The device to run the models on. Default is 'cpu'.

    Attributes:
    - device (torch.device): The device to run the models on.
    - esm_model (torch.nn.Module): The pretrained ESM-2 model.
    - alphabet (str): The alphabet used by the ESM-2 model.
    - clan_model (psalm_clan): The pretrained clan model.
    - fam_model (psalm_fam): The pretrained fam model.
    - fam_maps (dict): The mapping between family labels and indices.
    - clan_fam_matrix (torch.Tensor): The matrix representing the relationship between clans and families.
    - threshold (float): The threshold value for prediction confidence.

    Methods:
    - annotate(seq_list): Annotates protein sequences using the PSALM algorithm.

    """

    def __init__(self, clan_model_name, fam_model_name, device='cpu'):
        self.device = torch.device(device)

        # Load ESM-2 model
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()

        # Load clan model
        self.clan_model = psalm_clan.from_pretrained(clan_model_name)
        self.clan_model = self.clan_model.to(self.device)

        # Load fam model
        self.fam_model, self.fam_maps = psalm_fam.from_pretrained(fam_model_name)
        self.fam_model = self.fam_model.to(self.device)
        self.clan_fam_matrix = self.fam_maps['clan_family_matrix'].to(device)
    
    def read_fasta(self, fasta_file_path):
        """
        Reads a FASTA file and returns a list of tuples containing the sequence ID and sequence.

        Parameters:
        fasta_file_path (str): The path to the FASTA file.

        Returns:
        list: A list of tuples, where each tuple contains the sequence ID and sequence.
        """
        with open(fasta_file_path, "r") as handle:
            seq_list = [(record.id, str(record.seq)) for record in SeqIO.parse(handle, "fasta")]
        return seq_list

    def esm2_embed(self, seq_list):
        """
        Generate embeddings for the given data using the ESM model.

        Args:
            seq_list (list): A list of protein sequences.

        Returns:
            torch.Tensor: A tensor containing the token representations of the input sequences.
        """
        batch_converter = self.alphabet.get_batch_converter()
        _, _, batch_tokens = batch_converter(seq_list)
        device = next(self.esm_model.parameters()).device  # Use the same device as the model
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        token_representations = token_representations.squeeze()
        return token_representations
    
    def plot_sequences(self, seq_list,threshold,save_path):
        """
        Process a list of protein sequences.

        Args:
            seq_list (list): A list of tuples containing sequence names and sequences.
            threshold (float): The threshold value for prediction confidence.
            save_path (str): The path to save the visualization plots.

        Returns:
            tuple: A tuple containing the predictions for family and clan.

        """
        # Embed with ESM-2
        esm2_embeds = self.esm2_embed(seq_list)
        for i, seq_tup in enumerate(seq_list):
            seq_name = seq_tup[0]
            seq_len = len(seq_tup[1])
            if len(seq_list) == 1:
                seq_esm2_embed = esm2_embeds[:seq_len, :]
            else:
                seq_esm2_embed = esm2_embeds[i, :seq_len, :]
            clan_preds = self.clan_model(seq_esm2_embed)
            fam_preds = self.fam_model(seq_esm2_embed, clan_preds, self.clan_fam_matrix)
            # Visualize the predictions
            clan_preds = clan_preds.detach().cpu()
            fam_preds = fam_preds.detach().cpu()
            plot_predictions(fam_preds, clan_preds, self.fam_maps, threshold, seq_name,save_path=save_path)

    def annotate(self, seq_list, threshold=0.72,save_path=None):
        """
        Annotates protein sequences using PSALM.

        Parameters:
        - seq_list (list): A list of tuples, where each tuple contains the sequence name and sequence.
        - threshold (float, optional): The threshold value for prediction confidence. Default is 0.72.
        - save_path (str, optional): The path to save the visualization plots. Default is None.

        Returns:
        None
        """
        if save_path:
            if not save_path.endswith('/'):
                save_path = save_path+"/"
            os.makedirs(save_path, exist_ok=True)
        
        long_seq_list = []
        seq_list = [seq_tup for seq_tup in seq_list if len(seq_tup[1]) <= 4096 or long_seq_list.append(seq_tup)]

        self.plot_sequences(seq_list,threshold,save_path)

        if long_seq_list:
            warnings.warn("PSALM annotations may be unreliable for sequence lengths greater than 4096.")
            self.plot_sequences(long_seq_list,threshold,save_path)
    