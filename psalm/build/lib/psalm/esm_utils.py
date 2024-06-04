import torch
from torch import nn
import esm

class esm2_model(nn.Module):
    def __init__(self):
        super().__init__()  # Call the parent class's initializer first
        """Initialize the ESM model."""
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model.eval()

    def forward(self, data):
        """Generate embeddings for the given data using the ESM model."""
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        token_representations = token_representations.squeeze()
        return token_representations