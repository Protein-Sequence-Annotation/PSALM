import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import hmmscan_utils as hu
import pickle
from pathlib import Path
from esm import pretrained, FastaBatchedDataset
import sys

###########################################################
# Functions for data processing and model creation
###########################################################

def set_seeds(seed):

    """
    Set all seeds to the same value for reproducibility

    Args:
        seed (int): An integer that serves as the common seed
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return

##############################################################
# Dataset class to process sequences, create targets and batch
##############################################################

class DataUtils():

    def __init__(self, shard_path, num_shards, esm_model_name, limit, mode, device):

        with open(Path(shard_path) / 'maps.pkl', 'rb') as f:
            self.maps = pickle.load(f)
        
        self.clan_count = len(self.maps["clan_idx"])
        self.fam_count = len(self.maps["fam_idx"])
        self.shard_path = Path(shard_path)
        self.num_shards = num_shards
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_name)
        self.last_layer = self.esm_model.num_layers
        self.embedding_dim = self.esm_model.embed_dim
        self.length_limit = limit
        self.mode = mode

        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        self.esm_model.eval()
        self.esm_model.to(device)

    def __len__(self) -> int:

        """
        Get the number of shards in the dataset
        """

        return self.num_shards

    def __getitem__(self, idx: int) -> str:

        """
        Return the shard path at position idx

        Args:
            idx (int): The position of interest
        """

        return self.shard_path / f'{self.mode}_scan' / f'split_{idx}_{self.mode}_ids_full.fasta_scan.txt'

    def get_fasta(self, idx: int) -> str:

        """
        Return the fasta file for shard at position idx

        Args:
            idx (int): The position of interest
        """

        return self.shard_path / f'{self.mode}_fasta' / f'split_{idx}_{self.mode}_ids_full.fasta'

    def get_dataset(self, idx):
        
        """
        Return FastaBatchedDataset from the esm model

        Args:
            idx (int): Shard file index of interest

        Returns:
            dataset (Dataset): dataset with all sequences in shard
        """

        dataset = FastaBatchedDataset.from_file(self.shard_path / f'{self.mode}_fasta' / f'split_{idx}_{self.mode}_ids_full.fasta')

        return dataset

    def filter_batches(self, data, keys):

        """
        Temporary method to remove sequences with no hits in the hmmscan fasta file

        Args:
            data (Dataset): The current batch of sequences
            keys (Iterator): Iterator over sequences with hits

        Returns:
            data (Dataset): dataset with all no-hit entries removed
        """

        bad_idxs = []

        for idx, seq_name in enumerate(data.sequence_labels):
            seq_id = seq_name.split()[0]
            if seq_id not in keys:
                bad_idxs.append(idx) # This was originally .append(seq_id) but that isn't correct??

        data.sequence_strs = [x for i,x in enumerate(data.sequence_strs) if i not in bad_idxs]
        data.sequence_labels = [x for i,x in enumerate(data.sequence_labels) if i not in bad_idxs]

        return data

    def get_dataloader(self, data):

        """
        Return a data loader for the current dataset

        Args:
            data (Dataset): dataset to pad, batch and crop

        Returns:
            data_loader (DataLoader): a data loader with the appropriate padding function
        """

        seq_length = min(max(len(seq) for seq in data.sequence_strs), self.length_limit) # At most 'limit' length
        tokens_per_batch = self.length_limit
        batches = data.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

        data_loader = DataLoader(
                    data, 
                    collate_fn = self.alphabet.get_batch_converter(seq_length), 
                    batch_sampler = batches)
        
        return data_loader

    def parse_shard(self, idx) -> dict:

        """
        Parses a shard and returns hmmscan dict

        Args:
            idx (int): Shard file index of interest

        Returns:
            hmmscan_dict (Dict): A dictionary containing the results of a parsed hmmscan file
        """

        hmmscan_dict = hu.parse_hmmscan_results(self.shard_path / f'{self.mode}_scan' / f'split_{idx}_{self.mode}_ids_full.fasta_scan.txt')

        return hmmscan_dict

    def get_embedding(self, tokens):

        """
        Returns the esm embedding for the given sequence

        Args:

        Returns:
            embedding (torch.Tensor): tensor containing the embedding for the given sequence
        """

        return self.esm_model(tokens, repr_layers=[self.last_layer], return_contacts=False)

###########################################################
# Train step
###########################################################
 
def train_step(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a train step for one batch

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        loss_fn (nn.Loss): Loss function for the model
        optimizer (torch.optim.Optimizer): Optimizer for the classifier
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): dictionary with parsed results of hmmscan for current shard

    Returns:
        shard_loss (torch.Float32): loss over the entire shard
        shard_length (torch.Int): length of all sequences in shard
    """

    shard_loss = 0
    n_batches = len(data_loader)

    classifier.train()

    for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

        tokens = tokens.to(device)
        embedding = data_utils.get_embedding(tokens)
        
        batch_loss = torch.zeros(1, requires_grad=True).to(device)
        batch_lengths = 0

        optimizer.zero_grad()
        n_seqs = len(labels)

        sequence_labels = [x.split()[0] for x in labels]

        for idx, label in enumerate(sequence_labels):

            _, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
            
            stop_index = min(len(seqs[idx]), data_utils.length_limit)
            
            clan_vector = torch.tensor(clan_vector[:stop_index]).to(device) # clip the clan_vector to the truncated sequence length

            # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
            preds = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])

            # loss is averaged over the whole sequence
            loss = loss_fn(preds, clan_vector) 
            batch_loss = batch_loss + loss


        batch_loss = batch_loss / n_seqs # Average loss over all sequences in the batch
        batch_loss.backward()
        optimizer.step()

        shard_loss += batch_loss.item()

    return shard_loss, n_batches

def train_stepFamMoE(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a train step for one batch

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        loss_fn (nn.Loss): Loss function for the model
        optimizer (torch.optim.Optimizer): Optimizer for the classifier
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): dictionary with parsed results of hmmscan for current shard

    Returns:
        shard_loss (torch.Float32): loss over the entire shard
        shard_length (torch.Int): length of all sequences in shard
    """

    shard_loss = 0
    n_batches = len(data_loader)

    classifier.train()

    for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

        tokens = tokens.to(device)
        embedding = data_utils.get_embedding(tokens)
        
        batch_loss = torch.zeros(1, requires_grad=True).to(device)

        optimizer.zero_grad()
        n_seqs = len(labels)

        sequence_labels = [x.split()[0] for x in labels]

        for idx, label in enumerate(sequence_labels):

            fam_vector_raw, clan_vector = hu.generate_domain_position_list2(hmm_dict, label, data_utils.maps)
            
            stop_index = min(len(seqs[idx]), data_utils.length_limit)
            #print(clan_vector,stop_index)
            clan_vector = torch.tensor(clan_vector[:stop_index,:]).to(device) # clip the clan_vector to the truncated sequence length
            fam_vector = np.argmax(fam_vector_raw, axis=1)
            fam_vector = torch.tensor(fam_vector[:stop_index]).to(device) # clip the fam_vector to the truncated sequence length
            fam_vector_raw = torch.tensor(fam_vector_raw[:stop_index,:]).to(device)

            # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
            fam_preds = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_vector)

            # loss is averaged over the whole sequence
            fam_loss = F.cross_entropy(fam_preds, fam_vector) + 0.1*F.l1_loss(fam_preds, fam_vector_raw) #loss    

            batch_loss = batch_loss + fam_loss #loss

        
        batch_loss = batch_loss / n_seqs
        batch_loss.backward()
        optimizer.step()

        shard_loss += batch_loss.item()

    return shard_loss, n_batches 


def train_stepFamSimple(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a train step for one batch

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        loss_fn (nn.Loss): Loss function for the model
        optimizer (torch.optim.Optimizer): Optimizer for the classifier
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): dictionary with parsed results of hmmscan for current shard

    Returns:
        shard_loss (torch.Float32): loss over the entire shard
        shard_length (torch.Int): length of all sequences in shard
    """

    shard_loss = 0
    n_batches = len(data_loader)

    classifier.train()

    for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

        tokens = tokens.to(device)
        embedding = data_utils.get_embedding(tokens)
        
        batch_loss = torch.zeros(1, requires_grad=True).to(device)

        optimizer.zero_grad()
        n_seqs = len(labels)

        sequence_labels = [x.split()[0] for x in labels]

        for idx, label in enumerate(sequence_labels):

            fam_vector_raw, clan_vector = hu.generate_domain_position_list2(hmm_dict, label, data_utils.maps)
            
            stop_index = min(len(seqs[idx]), data_utils.length_limit)
            clan_vector = torch.tensor(clan_vector[:stop_index,:]).to(device) # clip the clan_vector to the truncated sequence length
            fam_vector = np.argmax(fam_vector_raw, axis=1)
            fam_vector = torch.tensor(fam_vector[:stop_index]).to(device) # clip the fam_vector to the truncated sequence length
            fam_vector_raw = torch.tensor(fam_vector_raw[:stop_index,:]).to(device) # clip the fam_vector to the truncated sequence length

            # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
            weighted_fam_preds,fam_preds = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_vector)
            # clan_fam_matrix = data_utils.maps['clan_family_matrix'].to(device)
            # clan_fam_weights = torch.matmul(clan_vector,clan_fam_matrix)

            # loss is averaged over the whole sequence

            # non_idr = fam_vector != 19632
            # non_idr_loss = F.cross_entropy(weighted_fam_preds[non_idr], fam_vector[non_idr]) if non_idr.sum() > 0 else 0.

            # fam_loss = non_idr_loss + 0.05*F.l1_loss(fam_preds, fam_vector_raw)         

            fam_loss = F.cross_entropy(weighted_fam_preds, fam_vector) + 0.05*F.l1_loss(fam_preds, fam_vector_raw)         

            batch_loss = batch_loss + fam_loss #loss

        
        batch_loss = batch_loss / n_seqs
        batch_loss.backward()
        optimizer.step()

        shard_loss += batch_loss.item()

    return shard_loss, n_batches 


def train_step_weighted(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a train step for one batch with weighted loss between IDR and other regions

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        loss_fn (nn.Loss): Loss function for the model
        optimizer (torch.optim.Optimizer): Optimizer for the classifier
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): dictionary with parsed results of hmmscan for current shard

    Returns:
        shard_loss (torch.Float32): loss over the entire shard
        shard_length (torch.Int): length of all sequences in shard
    """

    shard_loss = 0
    n_batches = len(data_loader)

    classifier.train()

    for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

        tokens = tokens.to(device)
        embedding = data_utils.get_embedding(tokens)
        
        batch_loss = torch.zeros(1, requires_grad=True).to(device)
        batch_lengths = 0

        optimizer.zero_grad()
        n_seqs = len(labels)

        sequence_labels = [x.split()[0] for x in labels]

        for idx, label in enumerate(sequence_labels):

            _, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
            
            stop_index = min(len(seqs[idx]), data_utils.length_limit)
            
            clan_vector = torch.tensor(clan_vector[:stop_index]).to(device) # clip the clan_vector to the truncated sequence length

            # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
            preds = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])

            # loss is averaged over the whole sequence
            idr = clan_vector == 656

            idr_loss = F.cross_entropy(preds[idr], clan_vector[idr]) if idr.sum() > 0 else 0.
            non_idr_loss = F.cross_entropy(preds[~idr], clan_vector[~idr]) if idr.sum() != idr.shape[0] else 0.

            total_loss = 0.25*idr_loss + 0.75*non_idr_loss
            batch_loss = batch_loss + total_loss

        batch_loss = batch_loss / n_seqs # Average loss over all sequences in the batch
        batch_loss.backward()
        optimizer.step()

        shard_loss += batch_loss.item()

    return shard_loss, n_batches

###########################################################
# Test step
###########################################################

def test_step(data_loader,
               classifier,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a test step for one batch

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): dictionary with parsed results of hmmscan for current shard

    Returns:
        all_preds (Dict): predictions for each sequence: top 2 values, indices, target
        shard_length (torch.Int): length of all sequences in shard
    """

    all_preds = {}

    n_batches = len(data_loader)

    classifier.eval()

    with torch.inference_mode():

        for batch_id, (labels, seq, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]

            for idx, label in enumerate(sequence_labels):

                _, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
                
                stop_index = min(len(seq[idx]), data_utils.length_limit)
                
                clan_vector = torch.tensor(clan_vector[:stop_index]).to(device) # clip the clan_vector to the truncated sequence length

                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                preds = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])
                preds = F.softmax(preds, dim=1)

                # Store predictions
                all_preds[label] = {}
                top_two_vals, top_two_indices = torch.topk(preds, k=2, dim=1)
                all_preds[label]['clan_vals'] = top_two_vals.cpu().numpy()
                all_preds[label]['clan_idx'] = top_two_indices.cpu().numpy()
                all_preds[label]['clan_true'] = clan_vector.cpu().numpy()

    return all_preds, n_batches

def test_stepFam(data_loader,
               classifier,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a test step for one batch

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): dictionary with parsed results of hmmscan for current shard

    Returns:
        all_preds (Dict): predictions for each sequence: top 2 values, indices, target
        shard_length (torch.Int): length of all sequences in shard
    """

    all_preds = {}

    n_batches = len(data_loader)

    classifier.eval()

    with torch.inference_mode():

        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]

            for idx, label in enumerate(sequence_labels):

                fam_vector, clan_vector = hu.generate_domain_position_list2(hmm_dict, label, data_utils.maps)
                
                stop_index = min(len(seqs[idx]), data_utils.length_limit)
                
                fam_vector = torch.tensor(fam_vector[:stop_index,:]).to(device) # clip the clan_vector to the truncated sequence length
                clan_vector = torch.tensor(clan_vector[:stop_index,:]).to(device)

                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                preds, _ = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_vector)
                preds = F.softmax(preds, dim=1)

                # Store predictions
                all_preds[label] = {}
                # top_two_vals, top_two_indices = torch.topk(preds, k=2, dim=1)
                # all_preds[label]['fam_vals'] = top_two_vals.cpu().numpy()
                # all_preds[label]['fam_idx'] = top_two_indices.cpu().numpy()
                # all_preds[label]['fam_true'] = fam_vector.cpu().numpy()
                # all_preds[label]['clan_true'] = clan_vector.cpu().numpy()
                all_preds[label]['fam_pred'] = preds.cpu().numpy()

    return all_preds, n_batches
