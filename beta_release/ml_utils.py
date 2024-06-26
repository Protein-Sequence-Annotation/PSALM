import numpy as np
import torch
from torch.utils.data import DataLoader
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

def set_torch_seeds(seed):

    """
    Set all seeds to the same value for reproducibility

    Args:
        seed (int): An integer that serves as the common seed
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return

##############################################################
# Dataset class to process sequences, create targets and batch
##############################################################

class DataUtils():

    def __init__(self, shard_path, num_shards, esm_model_name, limit, mode, device, alt_suffix=""):

        with open(Path('info_files') / 'maps.pkl', 'rb') as f:
            self.maps = pickle.load(f)
        self.shard_path = Path(shard_path)
        self.clan_count = len(self.maps["clan_idx"])
        self.fam_count = len(self.maps["fam_idx"])
        self.num_shards = num_shards
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_model.to(device)
        self.esm_model = torch.compile(self.esm_model)
        self.last_layer = self.esm_model.num_layers
        self.embedding_dim = self.esm_model.embed_dim
        self.length_limit = limit
        self.mode = mode
        self.alt_suffix = alt_suffix
        self.scan_path = self.shard_path / f'{self.mode}_scan{self.alt_suffix}'
        self.fasta_path = self.shard_path / f'{self.mode}_fasta{self.alt_suffix}'
        self.onehot_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        self.esm_model.eval()

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

        return self.scan_path / f'split_{idx}_{self.mode}_ids_full.fasta_scan.txt'

    def get_fasta(self, idx: int) -> str:

        """
        Return the fasta file for shard at position idx

        Args:
            idx (int): The position of interest
        """

        return self.fasta_path / f'split_{idx}_{self.mode}_ids_full.fasta'

    def get_dataset(self, idx):
        
        """
        Return FastaBatchedDataset from the esm model

        Args:
            idx (int): Shard file index of interest

        Returns:
            dataset (Dataset): dataset with all sequences in shard
        """

        dataset = FastaBatchedDataset.from_file(self.fasta_path / f'split_{idx}_{self.mode}_ids_full.fasta')

        return dataset
    
    def get_custom_dataset(self, fpath):
        
        """
        Return FastaBatchedDataset from the esm model

        Args:
            idx (int): Shard file index of interest

        Returns:
            dataset (Dataset): dataset with all sequences in shard
        """

        dataset = FastaBatchedDataset.from_file(fpath)

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
                bad_idxs.append(idx)

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

        hmmscan_dict = hu.parse_hmmscan_results(self.scan_path / f'split_{idx}_{self.mode}_ids_full.fasta_scan.txt')

        return hmmscan_dict

    def get_embedding(self, tokens):

        """
        Returns the esm embedding for the given sequence

        Args:

        Returns:
            embedding (torch.Tensor): tensor containing the embedding for the given sequence
        """

        # return self.esm_model(tokens, repr_layers=[self.last_layer], return_contacts=False)
        return self.esm_model(tokens, repr_layers=[self.last_layer], return_contacts=False)["representations"][self.last_layer]
    
    def get_onehots(self, seq):
            
            """
            Returns the one hot encoding for the given sequence
    
            Args:
                seq (str): sequence of interest
    
            Returns:
                onehot (torch.Tensor): tensor containing the one hot encoding of the sequence
            """
            
            onehot = torch.zeros(len(seq), len(self.onehot_alphabet))
            # Convert the sequence to a list of indices in the alphabet
            indices = [self.onehot_alphabet.find(char) for char in seq]
            # Convert the list of indices to a tensor
            indices_tensor = torch.tensor(indices)
            # Create a one-hot encoding using the indices tensor
            onehot = torch.nn.functional.one_hot(indices_tensor, num_classes=len(self.onehot_alphabet)).float()
            return onehot

###########################################################
# Train step
###########################################################
 
def train_step_clan(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a train step for one batch - trains clan prediction head

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
        n_batches (torch.Int): number of batches
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

            _, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
            
            stop_index = min(len(seqs[idx]), data_utils.length_limit)

            clan_vector = np.argmax(clan_vector, axis=1)
            
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

def train_step_clan_batch(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a train step for one batch - trains clan prediction head

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
        n_batches (torch.Int): number of batches
    """

    shard_loss = 0
    n_batches = len(data_loader)

    classifier.train()

    for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

        tokens = tokens.to(device)
        embedding = data_utils.get_embedding(tokens)

        optimizer.zero_grad()
        n_seqs = len(labels)

        sequence_labels = [x.split()[0] for x in labels]
        stop_indices = [min(len(seq), data_utils.length_limit) for seq in seqs]

        target_vectors = np.zeros((len(sequence_labels), embedding.shape[1]))
        mask = torch.zeros((embedding.shape[0], embedding.shape[1]))

        for idx, label in enumerate(sequence_labels):

            _, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
            clan_vector = np.argmax(clan_vector, axis=1)
        
            target_vectors[idx, 1:stop_indices[idx]+1] = clan_vector[:stop_indices[idx]]
            mask[idx, 1:stop_indices[idx]+1] = 1
        
        target_vectors = torch.tensor(target_vectors, dtype=torch.long).to(device)
        mask = mask.to(device)

        # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
        preds = classifier(embedding, mask)

        # loss is averaged over the whole sequence
        loss = loss_fn(preds[mask.bool()], target_vectors[mask.bool()]) / 1.0 # Average loss over all sequences in the batch

        # Full version - 12:05 min per shard
        # No generate_position - 11:57 min per shard
        # No masked loss - 12:09 min per shard
        # No backward pass - 10:17 min per shard
        # No prediction, just randn - 9:07 min per shard
        # No prediction, just ones - 9:00 min per shard
        # No mask/target creation - 11:12 min shard
        # No embedding, just 4096 ones - way too long
        # No embedding, no prediction - 3:07 mins
        # Torch compile default - 11:45 min per shard
        # mat mul change + erduce-overhead - 5:10 min per shard
        # max-autotune + matmul - 

        loss.backward()
        optimizer.step()

        shard_loss += loss.item()

    return shard_loss, n_batches

def train_step_fam(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a train step for one batch - trains family prediction head

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
        n_batches (torch.Int): number of batches
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

            fam_vector_raw, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
            
            stop_index = min(len(seqs[idx]), data_utils.length_limit)
            clan_vector = torch.tensor(clan_vector[:stop_index,:]).to(device) # clip the clan_vector to the truncated sequence length
            fam_vector = np.argmax(fam_vector_raw, axis=1)
            fam_vector = torch.tensor(fam_vector[:stop_index]).to(device) # clip the fam_vector to the truncated sequence length
            fam_vector_raw = torch.tensor(fam_vector_raw[:stop_index,:]).to(device) # clip the fam_vector to the truncated sequence length

            # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
            weighted_fam_preds,fam_preds = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_vector)      

            fam_loss = F.cross_entropy(weighted_fam_preds, fam_vector) #+ 0.05*F.l1_loss(fam_preds, fam_vector_raw)         

            batch_loss = batch_loss + fam_loss #loss

        
        batch_loss = batch_loss / n_seqs
        batch_loss.backward()
        optimizer.step()

        shard_loss += batch_loss.item()

    return shard_loss, n_batches


###########################################################
# Test step
###########################################################

def test_step_clan(data_loader,
               classifier,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a test step for one batch - only clan prediction head

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

    shard_preds = {}

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
                
                clan_vector = np.argmax(clan_vector, axis=1)
                clan_vector = torch.tensor(clan_vector[:stop_index]).to(device) # clip the clan_vector to the truncated sequence length, leave on cpu

                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                preds = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])
                preds = F.softmax(preds, dim=1)

                clan_preds, clan_idx = torch.topk(preds, k=1, dim=1)

                # Store predictions
                shard_preds[label] = {}

                shard_preds[label]['clan_preds'] = clan_preds.cpu().numpy()
                shard_preds[label]['clan_idx'] = clan_idx.cpu().numpy()
                shard_preds[label]['clan_true'] = clan_vector.numpy()

    return shard_preds, n_batches

def test_step_clan_batch(data_loader,
               classifier,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a test step for one batch - only clan prediction head

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

    shard_preds = {}

    n_batches = len(data_loader)

    classifier.eval()

    with torch.inference_mode():

        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]

            stop_indices = [min(len(seq), data_utils.length_limit) for seq in seqs]

            target_vectors = np.zeros((len(sequence_labels), embedding["representations"][data_utils.last_layer].shape[1]))
            mask = torch.zeros((embedding["representations"][data_utils.last_layer].shape[0], embedding["representations"][data_utils.last_layer].shape[1]))

            for idx, label in enumerate(sequence_labels):

                _, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
                clan_vector = np.argmax(clan_vector, axis=1)
            
                target_vectors[idx, 1:stop_indices[idx]+1] = clan_vector[:stop_indices[idx]]
                mask[idx, 1:stop_indices[idx]+1] = 1
            
            mask = mask.to(device)            

            # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
            preds = classifier(embedding["representations"][data_utils.last_layer], mask)
            
            preds = F.softmax(preds, dim=1) # We don't care about this softmax dimension for mask

            clan_preds, clan_idx = torch.topk(preds, k=1, dim=1) # This has same dimensions as mask

            # Store predictions
            

            for idx in range(len(stop_indices)):
                
                shard_preds[sequence_labels[idx]] = {}
                shard_preds[sequence_labels[idx]]['clan_preds'] = clan_preds[idx, 1:stop_indices[idx]+1].cpu().numpy()
                shard_preds[sequence_labels[idx]]['clan_idx'] = clan_idx[idx, 1:stop_indices[idx]+1].cpu().numpy()
                shard_preds[sequence_labels[idx]]['clan_true'] = target_vectors[idx, 1:stop_indices[idx]+1]

    return shard_preds, n_batches

def test_step_fam(data_loader,
               classifier_clan,
               classifier_fam,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a test step for one shard and only saves accuracy

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier_clan (nn.Module): The classifier head to decode esm embeddings for clan
        classifier_fam (nn.Module): The classifier head to decode esm embeddings for fam
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): dictionary with parsed results of hmmscan for current shard

    Returns:
        shard_preds (Dict): predictions for each sequence: top 2 accuracy, fam wise acc, adjusted acc
        shard_length (torch.Int): length of all sequences in shard
    """

    shard_preds = {}

    n_batches = len(data_loader)

    classifier_clan.eval()
    classifier_fam.eval()

    with torch.inference_mode():

        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]

            for idx, label in enumerate(sequence_labels):

                fam_vector, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
                
                stop_index = min(len(seqs[idx]), data_utils.length_limit)
                
                fam_vector = np.argmax(fam_vector[:stop_index,:], axis=1) # clip the clan_vector to the truncated sequence length, no need to place on gpu
                clan_vector = torch.tensor(clan_vector[:stop_index,:])

                clan_preds = classifier_clan(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])
                clan_preds = F.softmax(clan_preds, dim=1)
                
                # accessing the raw preds below
                _, fam_preds = classifier_fam(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_preds)

                clan_fam_weights = data_utils.maps['clan_family_matrix'].to(device)
                for i in range(clan_fam_weights.shape[0]): #shape is cxf 
                    indices = torch.nonzero(clan_fam_weights[i]).squeeze() #indices for the softmax
                    if i == 656:
                        fam_preds[:,indices] = 1 # IDR is 1:1 map
                    else:
                        fam_preds[:,indices] = torch.softmax(fam_preds[:,indices],dim=1)     

                # Multiply by clan, expand clan preds
                clan_preds_f = torch.matmul(clan_preds,clan_fam_weights)
                
                # element wise mul of preds and clan
                fam_preds = fam_preds * clan_preds_f

                # Store predictions
                shard_preds[label] = {}

                fam_preds, fam_idx = torch.topk(fam_preds, k=1, dim=1)
                clan_preds, clan_idx = torch.topk(clan_preds, k=1, dim=1)

                shard_preds[label]['fam_preds'] = fam_preds.cpu().numpy()
                shard_preds[label]['fam_idx'] = fam_idx.cpu().numpy()
                shard_preds[label]['clan_preds'] = clan_preds.cpu().numpy()
                shard_preds[label]['clan_idx'] = clan_idx.cpu().numpy()
                shard_preds[label]['fam_true'] = fam_vector.numpy()
                shard_preds[label]['clan_true'] = clan_vector.numpy()

    return shard_preds, n_batches

###########################################################
# Validation step
###########################################################

def validate_clan_batch(data_loader,
               classifier,
               loss_fn,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a validation step for one batch - only clan prediction head

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

    classifier.eval()

    with torch.inference_mode():

        shard_loss = 0

        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]
            stop_indices = [min(len(seq), data_utils.length_limit) for seq in seqs]

            target_vectors = np.zeros((len(sequence_labels), embedding.shape[1]))
            mask = torch.zeros((embedding.shape[0], embedding.shape[1]))

            for idx, label in enumerate(sequence_labels):

                _, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
                clan_vector = np.argmax(clan_vector, axis=1)
            
                target_vectors[idx, 1:stop_indices[idx]+1] = clan_vector[:stop_indices[idx]]
                mask[idx, 1:stop_indices[idx]+1] = 1
            
            target_vectors = torch.tensor(target_vectors, dtype=torch.long).to(device)
            mask = mask.to(device)

            # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
            preds = classifier(embedding, mask)

            # loss is averaged over the whole sequence
            loss = loss_fn(preds[mask.bool()], target_vectors[mask.bool()]) / 1.0 # Average loss over all sequences in the batch

            shard_loss += loss.item()

    return shard_loss