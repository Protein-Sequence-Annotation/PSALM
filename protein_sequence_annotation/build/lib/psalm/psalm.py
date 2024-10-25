import warnings
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import esm
from Bio import SeqIO
import pickle
import json
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import safetensors
from .viz_utils import plot_predictions

class psalm_clan(nn.Module, PyTorchModelHubMixin):
    """
    A PyTorch module for the PSALM Clan classifier model with batched inputs.

    Args:
        embed_dim (int): The embedding dimension from ESM-2 model.
        output_dim (int): The output dimension of PSALM-clan.

    Attributes:
        lstm (nn.LSTM): The BiLSTM layer.
        linear_stack (nn.Sequential): The sequential stack of linear layers.

    Methods:
        forward(x, mask): Performs forward pass through the model.
    """

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=True, batch_first=True)
        self.linear_stack = nn.Sequential(
            nn.Linear(2 * embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.LayerNorm(4 * embed_dim),
            nn.Linear(4 * embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.LayerNorm(4 * embed_dim),
            nn.Linear(4 * embed_dim, output_dim)
        )

    def forward(self, x, mask):
        x, _ = self.lstm(x)
        x = x * mask.unsqueeze(2)
        x = self.linear_stack(x)
        return x

    @classmethod
    def from_pretrained(cls, model_name, device):
        """
        Loads a pretrained model.

        Args:
            model_name (str): The name of the pretrained model.
            device (torch.device): The device to load the model onto.

        Returns:
            psalm_clan: The loaded model.
        """
        # Download the model's weights
        model_file = hf_hub_download(repo_id=f"{model_name}", filename="model.safetensors")
        # Load the model's weights onto CPU
        state_dict = safetensors.torch.load_file(model_file, device='cpu')

        # Load the model's configuration
        config_file = hf_hub_download(repo_id=f"{model_name}", filename="config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Create a new instance of the model class using the configuration and load the weights
        model = cls(config["input_dim"], config["output_dim"])
        model.load_state_dict(state_dict)
        # Move the model to the desired device
        model = model.to(device)

        return model


class psalm_fam(nn.Module, PyTorchModelHubMixin):
    """
    A PyTorch module for the PSALM Family classifier model with batched inputs.

    Args:
        embed_dim (int): The embedding dimension from the ESM-2 model.
        maps (dict): The mapping dictionaries including clan-family matrix.
        device (torch.device): The device to run the model on.

    Attributes:
        lstm (nn.LSTM): The bidirectional BiLSTM layer.
        linear_stack (nn.Sequential): The sequential stack of linear layers.
        clan_fam_matrix (torch.Tensor): The clan-family matrix.

    Methods:
        forward(x, mask, x_c): Performs forward pass of the model conditioned on clan predictions.
    """

    def __init__(self, embed_dim, maps, device):
        super().__init__()
        self.clan_fam_matrix = maps['clan_family_matrix'].to(device)
        self.f = self.clan_fam_matrix.shape[1]

        self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=True, batch_first=True)
        self.linear_stack = nn.Sequential(
            nn.Linear(2 * embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.LayerNorm(4 * embed_dim),
            nn.Linear(4 * embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.LayerNorm(4 * embed_dim),
            nn.Linear(4 * embed_dim, self.f)
        )

    def forward(self, x, mask, x_c):
        # Generate clan to fam weights
        weights = torch.matmul(x_c, self.clan_fam_matrix)

        # Get fam logits
        x, _ = self.lstm(x)
        x = x * mask.unsqueeze(2)
        x = self.linear_stack(x)

        # Element-wise multiplication of weights and fam_vectors
        weighted_fam_vectors = x * weights

        return weighted_fam_vectors, x

    @classmethod
    def from_pretrained(cls, model_name, device):
        """
        Loads a pretrained model.

        Args:
            model_name (str): The name of the pretrained model.
            device (torch.device): The device to load the model onto.

        Returns:
            Tuple[psalm_fam, Dict]: A tuple containing the loaded model and the maps dictionary.
        """
        # Download the model's weights
        model_file = hf_hub_download(repo_id=f"{model_name}", filename="model.safetensors")
        # Load the model's weights onto CPU
        state_dict = safetensors.torch.load_file(model_file, device='cpu')

        # Load the model's configuration
        config_file = hf_hub_download(repo_id=f"{model_name}", filename="config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Download the maps dictionary
        maps_file = hf_hub_download(repo_id=f"{model_name}", filename="maps.pkl")
        with open(maps_file, 'rb') as f:
            maps = pickle.load(f)

        # Create a new instance of the model class using the configuration and load the weights
        model = cls(config["input_dim"], maps, device)
        model.load_state_dict(state_dict)
        # Move the model to the desired device
        model = model.to(device)

        return model, maps


class psalm:
    """
    The `psalm` class is used for protein annotation using the PSALM algorithm with batched inputs.

    Parameters:
    - clan_model_name (str): The name of the pretrained clan model.
    - fam_model_name (str): The name of the pretrained fam model.
    - device (str, optional): The device to run the models on. Default is 'cpu'.

    Attributes:
    - device (torch.device): The device to run the models on.
    - esm_model (torch.nn.Module): The pretrained ESM-2 model.
    - alphabet (Alphabet): The alphabet used by the ESM-2 model.
    - clan_model (psalm_clan): The pretrained clan model.
    - fam_model (psalm_fam): The pretrained fam model.
    - fam_maps (dict): The mapping between family labels and indices.
    - clan_fam_matrix (torch.Tensor): The matrix representing the relationship between clans and families.

    Methods:
    - annotate(seq_list, batch_size=16, threshold=0.72, save_path=None, verbose=False): Annotates protein sequences using the PSALM algorithm.
    """

    def __init__(self, clan_model_name, fam_model_name, device='cpu'):
        self.device = torch.device(device)

        # Load ESM-2 model
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()

        # Load clan model
        self.clan_model = psalm_clan.from_pretrained(clan_model_name, self.device)
        self.clan_model = self.clan_model.to(self.device)
        self.clan_model.eval()

        # Load fam model and maps
        self.fam_model, self.fam_maps = psalm_fam.from_pretrained(fam_model_name, self.device)
        self.fam_model = self.fam_model.to(self.device)
        self.fam_model.eval()

        self.clan_fam_matrix = self.fam_maps['clan_family_matrix'].to(self.device)

        # Batch converter and tokens per batch
        self.batch_converter = self.alphabet.get_batch_converter()
        self.tokens_per_batch = 4096  # Adjust as needed

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

    def annotate(self, seq_list, batch_size=16, threshold=0.72, save_path=None, verbose=False):
        """
        Annotates protein sequences using PSALM.

        Parameters:
        - seq_list (list): A list of tuples, where each tuple contains the sequence name and sequence.
        - batch_size (int, optional): The batch size for processing sequences. Default is 16.
        - threshold (float, optional): The threshold value for prediction confidence. Default is 0.72.
        - save_path (str, optional): The path to save the visualization plots. Default is None.
        - verbose (bool, optional): If True, print additional information. Default is False.

        Returns:
        None
        """
        if save_path:
            if not save_path.endswith('/'):
                save_path = save_path + "/"
            os.makedirs(save_path, exist_ok=True)

        # Split sequences into batches
        num_batches = (len(seq_list) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch_seqs = seq_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            # Process batch
            self._process_batch(batch_seqs, threshold, save_path, verbose)

    def _process_batch(self, seq_list, threshold, save_path, verbose):
        """
        Process a batch of sequences.

        Args:
            seq_list (list): A list of tuples containing sequence names and sequences.
            threshold (float): The threshold value for prediction confidence.
            save_path (str): The path to save the visualization plots.
            verbose (bool): If True, print additional information.

        Returns:
            None
        """
        # Convert sequences to tokens
        labels, strs, tokens = self.batch_converter(seq_list)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            # Get embeddings
            results = self.esm_model(tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            # Create mask
            batch_size, seq_len, _ = token_representations.size()
            mask = tokens != self.alphabet.padding_idx
            mask = mask[:, 1:-1]  # Remove start + end tokens
            mask = mask.to(self.device)
            # Remove start and end tokens from embeddings
            token_representations = token_representations[:, 1:-1, :]
            # Run clan model
            clan_logits = self.clan_model(token_representations, mask)
            # Apply softmax to clan predictions
            clan_preds = F.softmax(clan_logits, dim=2)
            # Run fam model
            fam_preds, fam_logits = self.fam_model(token_representations, mask, clan_preds)
            # Clan-based normalization
            for i in range(self.clan_fam_matrix.shape[0]):  # shape is cxf
                indices = torch.nonzero(self.clan_fam_matrix[i]).squeeze()
                if i == self.clan_fam_matrix.shape[0] - 1:
                    fam_preds[:, :, indices] = 1  # IDR is 1:1 map
                else:
                    fam_preds[:, :, indices] = F.softmax(fam_preds[:, :, indices], dim=2)
            # Multiply by clan, expand clan preds
            clan_preds_f = torch.matmul(clan_preds, self.clan_fam_matrix)
            fam_preds = fam_preds * clan_preds_f
            # Process and visualize predictions
            for i, label in enumerate(labels):
                seq_name = label.split()[0]
                seq_len = len(seq_list[i][1])
                seq_clan_preds = clan_preds[i, :seq_len, :].detach().cpu()
                seq_fam_preds = fam_preds[i, :seq_len, :].detach().cpu()
                if verbose and save_path is not None:
                    # Save clan and fam preds
                    with open(save_path + seq_name + "_clan_preds.pkl", "wb") as f:
                        pickle.dump(seq_clan_preds, f)
                    with open(save_path + seq_name + "_fam_preds.pkl", "wb") as f:
                        pickle.dump(seq_fam_preds, f)
                plot_predictions(seq_fam_preds, seq_clan_preds, self.fam_maps, threshold, seq_name, save_path=save_path)