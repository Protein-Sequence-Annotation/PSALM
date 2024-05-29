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

        with open(Path(shard_path) / 'maps.pkl', 'rb') as f:
            self.maps = pickle.load(f)
        self.shard_path = Path(shard_path)
        self.clan_count = len(self.maps["clan_idx"])
        self.fam_count = len(self.maps["fam_idx"])
        self.num_shards = num_shards
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_name)
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

        return self.esm_model(tokens, repr_layers=[self.last_layer], return_contacts=False)
    
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

def train_stepClanFinetune(data_loader,
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

            ## make clan vector
            if label.split()[0] == "Shuffled":
                clan_idx = data_utils.maps["clan_idx"]["IDR"]

            else:
                clan_idx = data_utils.maps["clan_idx"][label.split()[0].split("_")[-1]] ### Look at the finetune shards if confused

            clan_vector = torch.full((len(seqs[idx]),), clan_idx).to(device)
            # print(label, clan_idx, clan_vector)
            
            stop_index = min(clan_vector.shape[0], data_utils.length_limit)
            
            clan_vector = clan_vector[:stop_index] # clip the clan_vector to the truncated sequence length

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

            fam_loss = F.cross_entropy(weighted_fam_preds, fam_vector) #+ 0.05*F.l1_loss(fam_preds, fam_vector_raw)         

            batch_loss = batch_loss + fam_loss #loss

        
        batch_loss = batch_loss / n_seqs
        batch_loss.backward()
        optimizer.step()

        shard_loss += batch_loss.item()

    return shard_loss, n_batches

def train_stepClanFamSimple(data_loader,
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

            # Not teacher forcing at the moment - might be dumb, but just interested in seeing output
            clan_preds, weighted_fam_preds, fam_preds = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])     

            fam_loss = F.cross_entropy(weighted_fam_preds, fam_vector) #+ 0.05*F.l1_loss(fam_preds, fam_vector_raw) # Can decide L1
            clan_loss = F.cross_entropy(clan_preds, clan_vector) #+ 0.05*F.l1_loss(clan_preds, clan_vector) # Can decide L1

            batch_loss = batch_loss + fam_loss + clan_loss # loss for each - check magnitude?

        batch_loss = batch_loss / n_seqs
        batch_loss.backward()
        optimizer.step()

        shard_loss += batch_loss.item()

    return shard_loss, n_batches

def train_stepFamOneHot(data_loader,
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

            # Generate one hot embedding
            embedding = data_utils.get_onehots(seqs[idx][:stop_index]).to(device)
            

            # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
            weighted_fam_preds,fam_preds = classifier(embedding, clan_vector)      

            fam_loss = F.cross_entropy(weighted_fam_preds, fam_vector) #+ 0.05*F.l1_loss(fam_preds, fam_vector_raw)         

            batch_loss = batch_loss + fam_loss #loss

        
        batch_loss = batch_loss / n_seqs
        batch_loss.backward()
        optimizer.step()

        shard_loss += batch_loss.item()

    return shard_loss, n_batches

def train_stepClanOneHot(data_loader,
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
        
        batch_loss = torch.zeros(1, requires_grad=True).to(device)

        optimizer.zero_grad()
        n_seqs = len(labels)

        sequence_labels = [x.split()[0] for x in labels]

        for idx, label in enumerate(sequence_labels):

            _, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
            
            stop_index = min(len(seqs[idx]), data_utils.length_limit)
            
            clan_vector = torch.tensor(clan_vector[:stop_index]).to(device) # clip the clan_vector to the truncated sequence length

            
            # get clan one hot embedding
            embedding = data_utils.get_onehots(seqs[idx][:stop_index]).to(device)
            # print(len(seqs[idx]))
            # print(embedding.shape)

            # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
            preds = classifier(embedding)

            # loss is averaged over the whole sequence
            loss = loss_fn(preds, clan_vector) 
            batch_loss = batch_loss + loss


        batch_loss = batch_loss / n_seqs # Average loss over all sequences in the batch
        batch_loss.backward()
        optimizer.step()

        shard_loss += batch_loss.item()

    return shard_loss, n_batches


###########################################################
# Test step
###########################################################

def test_stepClan(data_loader,
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
                
                clan_vector = clan_vector[:stop_index] # clip the clan_vector to the truncated sequence length, leave on cpu

                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                preds = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])
                preds = F.softmax(preds, dim=1).cpu()

                # Store predictions
                shard_preds[label] = {}

                top_two_vals, top_two_indices = torch.topk(preds, k=2, dim=1)
                top_two_indices = top_two_indices.numpy() # convert vals as well, once we use it
                
                first = top_two_indices[:,0] == clan_vector
                second = top_two_indices[:,1] == clan_vector

                shard_preds[label]['top'] = first.mean()
                shard_preds[label]['top2'] = (first+second).mean()

                clans = np.unique(clan_vector)
                # pred_unique = np.unique(top_two_indices) # Unique from top? Or top 2?
                
                # Not implementing set wise match currently
                # common = np.intersect1d(fams, pred_unique)
                # shard_preds[label]['set_acc'] = common/fams.shape[0] # fraction of matching fams
                # shard_preds[label]['set_strict'] = (fams.shape[0] == pred_unique.shape[0]) and common.shape[0] == fams.shape[0]

                top_total = 0. # Clan wise top
                top2_total = 0. # Clan wise top 2

                for clan in clans:
                    idx = (clan_vector == clan)
                    first = top_two_indices[:,0][idx] == clan_vector[idx]
                    second = top_two_indices[:,1][idx] == clan_vector[idx]
                    
                    top_total += first.mean()
                    top2_total += (first+second).mean()

                shard_preds[label]['clan_top'] = (top_total / clans.shape[0])
                shard_preds[label]['clan_top2'] = (top2_total / clans.shape[0])
                
                adjusted_score = 0.
                dubious_pos = 0
                # If positions match, then full score
                # If within a 10 residue window (+5, -5), then full score
                # If confusion between NC and IDR, ignore

                for i in range(clan_vector.shape[0]):
                    if clan_vector[i] == top_two_indices[i,0]:
                        adjusted_score += 1
                    elif (clan_vector[i] == top_two_indices[max(0,i-5):i+1,0]).any() or \
                        (clan_vector[i] == top_two_indices[i:min(i+5,clan_vector.shape[0]),0]).any():
                        adjusted_score += 1
                    elif clan_vector[i] == 656 and top_two_indices[i,0] == 655 and top_two_indices[i,1] == 656:
                        dubious_pos += 1
                    elif clan_vector[i] == 655 and top_two_indices[i,0] == 656 and top_two_indices[i,1] == 655:
                        dubious_pos += 1

                shard_preds[label]['adjusted_acc'] = adjusted_score / (clan_vector.shape[0]-dubious_pos+1e-3)

                non_idr = clan_vector != 656
                first_non_idr = top_two_indices[:,0][non_idr] == clan_vector[non_idr]
                second_non_idr = top_two_indices[:,1][non_idr] == clan_vector[non_idr]

                shard_preds[label]['non_idr_top'] = first_non_idr.mean()
                shard_preds[label]['non_idr_top2'] = (first_non_idr+second_non_idr).mean()

    return shard_preds, n_batches

def test_stepFam(data_loader,
               classifier,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a test step for one shard and only saves accuracy

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): dictionary with parsed results of hmmscan for current shard

    Returns:
        shard_preds (Dict): predictions for each sequence: top 2 accuracy, fam wise acc, adjusted acc
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

            for idx, label in enumerate(sequence_labels):

                fam_vector, clan_vector = hu.generate_domain_position_list2(hmm_dict, label, data_utils.maps)
                
                stop_index = min(len(seqs[idx]), data_utils.length_limit)
                
                fam_vector = np.argmax(fam_vector[:stop_index,:], axis=1) # clip the clan_vector to the truncated sequence length, no need to place on gpu
                clan_vector = torch.tensor(clan_vector[:stop_index,:]).to(device) # Used in fam prediction, so send to gpu

                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                preds, _ = classifier(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_vector)
                preds = F.softmax(preds, dim=1).cpu()

                # Store predictions
                shard_preds[label] = {}

                top_two_vals, top_two_indices = torch.topk(preds, k=2, dim=1)
                top_two_indices = top_two_indices.numpy() # convert vals as well, once we use it
                
                first = top_two_indices[:,0] == fam_vector
                second = top_two_indices[:,1] == fam_vector

                shard_preds[label]['top'] = first.mean()
                shard_preds[label]['top2'] = (first+second).mean()

                fams = np.unique(fam_vector)
                # pred_unique = np.unique(top_two_indices) # Unique from top? Or top 2?
                
                # Not implementing set wise match currently
                # common = np.intersect1d(fams, pred_unique)
                # shard_preds[label]['set_acc'] = common/fams.shape[0] # fraction of matching fams
                # shard_preds[label]['set_strict'] = (fams.shape[0] == pred_unique.shape[0]) and common.shape[0] == fams.shape[0]

                top_total = 0. # Fam wise top
                top2_total = 0. # Fam wise top 2

                for fam in fams:
                    idx = (fam_vector == fam)
                    first = top_two_indices[:,0][idx] == fam_vector[idx]
                    second = top_two_indices[:,1][idx] == fam_vector[idx]
                    
                    top_total += first.mean()
                    top2_total += (first+second).mean()

                shard_preds[label]['fam_top'] = (top_total / fams.shape[0])
                shard_preds[label]['fam_top2'] = (top2_total / fams.shape[0])
                
                adjusted_score = 0.
                dubious_pos = 0
                # If positions match, then full score
                # If within a 10 residue window (+5, -5), then full score
                # Not implementing confusion between NC and IDR since this is fam and there is no NC

                for i in range(fam_vector.shape[0]):
                    if fam_vector[i] == top_two_indices[i,0]:
                        adjusted_score += 1
                    elif (fam_vector[i] == top_two_indices[max(0,i-5):i+1,0]).any() or \
                        (fam_vector[i] == top_two_indices[i:min(i+5,fam_vector.shape[0]),0]).any():
                        adjusted_score += 1

                shard_preds[label]['adjusted_acc'] = adjusted_score / (fam_vector.shape[0]-dubious_pos+1e-3)

                non_idr = fam_vector != 19632
                first_non_idr = top_two_indices[:,0][non_idr] == fam_vector[non_idr]
                second_non_idr = top_two_indices[:,1][non_idr] == fam_vector[non_idr]

                shard_preds[label]['non_idr_top'] = first_non_idr.mean()
                shard_preds[label]['non_idr_top2'] = (first_non_idr+second_non_idr).mean()

    return shard_preds, n_batches

def test_stepFamJoint(data_loader,
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

                fam_vector, clan_vector = hu.generate_domain_position_list2(hmm_dict, label, data_utils.maps)
                
                stop_index = min(len(seqs[idx]), data_utils.length_limit)
                
                fam_vector = np.argmax(fam_vector[:stop_index,:], axis=1) # clip the clan_vector to the truncated sequence length, no need to place on gpu
                clan_vector = torch.tensor(clan_vector[:stop_index,:]).to(device) # Used in fam prediction, so send to gpu

                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                clan_preds = classifier_clan(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])
                clan_preds = F.softmax(clan_preds, dim=1)
                
                ########
                ##ONEHOT
                # # Find the indices of the max values along dimension 1
                # max_indices = torch.argmax(clan_preds, dim=1)

                # # Create a tensor of zeros with the same shape as clan_preds
                # clan_preds_one_hot = torch.zeros_like(clan_preds)

                # # Replace the max indices with 1
                # clan_preds_one_hot.scatter_(1, max_indices.unsqueeze(1), 1)
                # clan_preds = clan_preds_one_hot
                #########
                
                # preds, _ = classifier_fam(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_preds)
                # accessing the raw preds below
                _, preds = classifier_fam(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_preds)

                #########
                ##CLAN_CHUNK_FAM
                clan_fam_weights = data_utils.maps['clan_family_matrix'].to(device)
                # print(torch.nonzero(clan_fam_weights[-1]).squeeze())
                for i in range(clan_fam_weights.shape[0]): #shape is cxf 
                    indices = torch.nonzero(clan_fam_weights[i]).squeeze() #indices for the softmax
                    if i == 656:
                        preds[:,indices] = 1 # IDR is 1:1 map
                    else:
                        preds[:,indices] = torch.softmax(preds[:,indices],dim=1)
                
                #########

                #########
                # Multiply by clan
                # expand clan preds
                clan_preds_f = torch.matmul(clan_preds,clan_fam_weights)
                # element wise mul of preds and clan
                preds = preds * clan_preds_f
                #########
                preds = preds.cpu()
                # preds = F.softmax(preds, dim=1).cpu()

                # Store predictions
                shard_preds[label] = {}

                top_two_vals, top_two_indices = torch.topk(preds, k=2, dim=1)
                top_two_indices = top_two_indices.numpy() # convert vals as well, once we use it
                
                first = top_two_indices[:,0] == fam_vector
                second = top_two_indices[:,1] == fam_vector

                shard_preds[label]['top'] = first.mean()
                shard_preds[label]['top2'] = (first+second).mean()

                fams = np.unique(fam_vector)
                # pred_unique = np.unique(top_two_indices) # Unique from top? Or top 2?
                
                # Not implementing set wise match currently
                # common = np.intersect1d(fams, pred_unique)
                # shard_preds[label]['set_acc'] = common/fams.shape[0] # fraction of matching fams
                # shard_preds[label]['set_strict'] = (fams.shape[0] == pred_unique.shape[0]) and common.shape[0] == fams.shape[0]

                top_total = 0. # Fam wise top
                top2_total = 0. # Fam wise top 2

                for fam in fams:
                    idx = (fam_vector == fam)
                    first = top_two_indices[:,0][idx] == fam_vector[idx]
                    second = top_two_indices[:,1][idx] == fam_vector[idx]
                    
                    top_total += first.mean()
                    top2_total += (first+second).mean()

                shard_preds[label]['fam_top'] = (top_total / fams.shape[0])
                shard_preds[label]['fam_top2'] = (top2_total / fams.shape[0])
                
                adjusted_score = 0.
                dubious_pos = 0
                # If positions match, then full score
                # If within a 10 residue window (+5, -5), then full score
                # Not implementing confusion between NC and IDR since this is fam and there is no NC

                for i in range(fam_vector.shape[0]):
                    if fam_vector[i] == top_two_indices[i,0]:
                        adjusted_score += 1
                    elif (fam_vector[i] == top_two_indices[max(0,i-5):i+1,0]).any() or \
                        (fam_vector[i] == top_two_indices[i:min(i+5,fam_vector.shape[0]),0]).any():
                        adjusted_score += 1

                shard_preds[label]['adjusted_acc'] = adjusted_score / (fam_vector.shape[0]-dubious_pos+1e-3)

                non_idr = fam_vector != 19632
                first_non_idr = top_two_indices[:,0][non_idr] == fam_vector[non_idr]
                second_non_idr = top_two_indices[:,1][non_idr] == fam_vector[non_idr]

                shard_preds[label]['non_idr_top'] = first_non_idr.mean()
                shard_preds[label]['non_idr_top2'] = (first_non_idr+second_non_idr).mean()

    return shard_preds, n_batches


def test_stepClanOneHot(data_loader,
               classifier,
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

    n_batches = len(data_loader)
    shard_preds = {}

    classifier.eval()
    with torch.inference_mode():

        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):
            
            batch_loss = torch.zeros(1, requires_grad=True).to(device)

            n_seqs = len(labels)

            sequence_labels = [x.split()[0] for x in labels]

            for idx, label in enumerate(sequence_labels):

                _, clan_vector = hu.generate_domain_position_list(hmm_dict, label, data_utils.maps)
                
                stop_index = min(len(seqs[idx]), data_utils.length_limit)
                
                clan_vector = clan_vector[:stop_index]
                
                # get clan one hot embedding
                embedding = data_utils.get_onehots(seqs[idx][:stop_index]).to(device)
                # print(len(seqs[idx]))
                # print(embedding.shape)

                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                preds = classifier(embedding)
                preds = preds.cpu()

                # Store predictions
                shard_preds[label] = {}

                top_two_vals, top_two_indices = torch.topk(preds, k=2, dim=1)
                top_two_indices = top_two_indices.numpy() # convert vals as well, once we use it
                
                first = top_two_indices[:,0] == clan_vector
                second = top_two_indices[:,1] == clan_vector

                shard_preds[label]['top'] = first.mean()
                shard_preds[label]['top2'] = (first+second).mean()

                clans = np.unique(clan_vector)
                # pred_unique = np.unique(top_two_indices) # Unique from top? Or top 2?
                
                # Not implementing set wise match currently
                # common = np.intersect1d(fams, pred_unique)
                # shard_preds[label]['set_acc'] = common/fams.shape[0] # fraction of matching fams
                # shard_preds[label]['set_strict'] = (fams.shape[0] == pred_unique.shape[0]) and common.shape[0] == fams.shape[0]

                top_total = 0. # Clan wise top
                top2_total = 0. # Clan wise top 2

                for clan in clans:
                    idx = (clan_vector == clan)
                    first = top_two_indices[:,0][idx] == clan_vector[idx]
                    second = top_two_indices[:,1][idx] == clan_vector[idx]
                    
                    top_total += first.mean()
                    top2_total += (first+second).mean()

                shard_preds[label]['clan_top'] = (top_total / clans.shape[0])
                shard_preds[label]['clan_top2'] = (top2_total / clans.shape[0])
                
                adjusted_score = 0.
                dubious_pos = 0
                # If positions match, then full score
                # If within a 10 residue window (+5, -5), then full score
                # If confusion between NC and IDR, ignore

                for i in range(clan_vector.shape[0]):
                    if clan_vector[i] == top_two_indices[i,0]:
                        adjusted_score += 1
                    elif (clan_vector[i] == top_two_indices[max(0,i-5):i+1,0]).any() or \
                        (clan_vector[i] == top_two_indices[i:min(i+5,clan_vector.shape[0]),0]).any():
                        adjusted_score += 1
                    elif clan_vector[i] == 656 and top_two_indices[i,0] == 655 and top_two_indices[i,1] == 656:
                        dubious_pos += 1
                    elif clan_vector[i] == 655 and top_two_indices[i,0] == 656 and top_two_indices[i,1] == 655:
                        dubious_pos += 1

                shard_preds[label]['adjusted_acc'] = adjusted_score / (clan_vector.shape[0]-dubious_pos+1e-3)

                non_idr = clan_vector != 656
                first_non_idr = top_two_indices[:,0][non_idr] == clan_vector[non_idr]
                second_non_idr = top_two_indices[:,1][non_idr] == clan_vector[non_idr]

                shard_preds[label]['non_idr_top'] = first_non_idr.mean()
                shard_preds[label]['non_idr_top2'] = (first_non_idr+second_non_idr).mean()

    return shard_preds, n_batches

def test_stepFamOneHot(data_loader,
               classifier,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a test step for one batch

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

    n_batches = len(data_loader)
    shard_preds = {}

    classifier.eval()
    with torch.inference_mode():

        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):
            
            batch_loss = torch.zeros(1, requires_grad=True).to(device)

            n_seqs = len(labels)

            sequence_labels = [x.split()[0] for x in labels]

            for idx, label in enumerate(sequence_labels):

                fam_vector_raw, clan_vector = hu.generate_domain_position_list2(hmm_dict, label, data_utils.maps)
                
                stop_index = min(len(seqs[idx]), data_utils.length_limit)
                clan_vector = torch.tensor(clan_vector[:stop_index,:]).to(device) # clip the clan_vector to the truncated sequence length
                fam_vector = np.argmax(fam_vector_raw, axis=1)
                fam_vector = fam_vector[:stop_index] # clip the fam_vector to the truncated sequence length

                # Generate one hot embedding
                embedding = data_utils.get_onehots(seqs[idx][:stop_index]).to(device)
                
                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                preds, _ = classifier(embedding, clan_vector)
                preds = preds.cpu()

                shard_preds[label] = {}

                top_two_vals, top_two_indices = torch.topk(preds, k=2, dim=1)
                top_two_indices = top_two_indices.numpy() # convert vals as well, once we use it
                
                first = top_two_indices[:,0] == fam_vector
                second = top_two_indices[:,1] == fam_vector

                shard_preds[label]['top'] = first.mean()
                shard_preds[label]['top2'] = (first+second).mean()

                fams = np.unique(fam_vector)

                top_total = 0. # Fam wise top
                top2_total = 0. # Fam wise top 2

                for fam in fams:
                    idx = (fam_vector == fam)
                    first = top_two_indices[:,0][idx] == fam_vector[idx]
                    second = top_two_indices[:,1][idx] == fam_vector[idx]
                    
                    top_total += first.mean()
                    top2_total += (first+second).mean()

                shard_preds[label]['fam_top'] = (top_total / fams.shape[0])
                shard_preds[label]['fam_top2'] = (top2_total / fams.shape[0])
                
                adjusted_score = 0.
                dubious_pos = 0
                # If positions match, then full score
                # If within a 10 residue window (+5, -5), then full score
                # Not implementing confusion between NC and IDR since this is fam and there is no NC

                for i in range(fam_vector.shape[0]):
                    if fam_vector[i] == top_two_indices[i,0]:
                        adjusted_score += 1
                    elif (fam_vector[i] == top_two_indices[max(0,i-5):i+1,0]).any() or \
                        (fam_vector[i] == top_two_indices[i:min(i+5,fam_vector.shape[0]),0]).any():
                        adjusted_score += 1

                shard_preds[label]['adjusted_acc'] = adjusted_score / (fam_vector.shape[0]-dubious_pos+1e-3)

                non_idr = fam_vector != 19632
                first_non_idr = top_two_indices[:,0][non_idr] == fam_vector[non_idr]
                second_non_idr = top_two_indices[:,1][non_idr] == fam_vector[non_idr]

                shard_preds[label]['non_idr_top'] = first_non_idr.mean()
                shard_preds[label]['non_idr_top2'] = (first_non_idr+second_non_idr).mean()

    return shard_preds, n_batches

def test_stepJointPreds(data_loader,
               classifier_clan,
               classifier_fam,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a test step for one shard and saves argmax prediction

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
                
                stop_index = min(len(seqs[idx]), data_utils.length_limit)

                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                clan_preds = classifier_clan(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])
                clan_preds = F.softmax(clan_preds, dim=1)
                         
                fam_preds, _ = classifier_fam(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_preds)
                fam_preds = F.softmax(fam_preds, dim=1)

                # Store predictions
                shard_preds[label] = {}

                fam_preds, fam_idx = torch.topk(fam_preds, k=1, dim=1)
                clan_preds, clan_idx = torch.topk(clan_preds, k=1, dim=1)

                shard_preds[label]['fam_preds'] = fam_preds.cpu().numpy()
                shard_preds[label]['fam_idx'] = fam_idx.cpu().numpy()
                shard_preds[label]['clan_preds'] = clan_preds.cpu().numpy()
                shard_preds[label]['clan_idx'] = clan_idx.cpu().numpy()
                
    return shard_preds, n_batches

def test_stepNegatives(data_loader,
               classifier_clan,
               classifier_fam,
               device,
               data_utils,
               hmm_dict):

    """
    Runs a test step for one shard and saves argmax prediction

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
                
                stop_index = min(len(seqs[idx]), data_utils.length_limit)

                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                clan_preds = classifier_clan(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])
                clan_preds = F.softmax(clan_preds, dim=1)
                         
                _, preds = classifier_fam(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_preds)
                
                # chunk fam clan
                clan_fam_weights = data_utils.maps['clan_family_matrix'].to(device)
                # print(torch.nonzero(clan_fam_weights[-1]).squeeze())
                for i in range(clan_fam_weights.shape[0]): #shape is cxf 
                    indices = torch.nonzero(clan_fam_weights[i]).squeeze() #indices for the softmax
                    if i == 656:
                        preds[:,indices] = 1 # IDR is 1:1 map
                    else:
                        preds[:,indices] = torch.softmax(preds[:,indices],dim=1)
                
                #Multiply by clan
                # expand clan preds
                clan_preds_f = torch.matmul(clan_preds,clan_fam_weights)
                # element wise mul of preds and clan
                preds = preds * clan_preds_f
                #########
                preds = preds.cpu()

                # Store predictions
                shard_preds[label] = {}

                fam_preds, fam_idx = torch.topk(preds, k=1, dim=1)
                clan_preds, clan_idx = torch.topk(clan_preds, k=1, dim=1)

                shard_preds[label]['fam_preds'] = fam_preds.cpu().numpy()
                shard_preds[label]['fam_idx'] = fam_idx.cpu().numpy()
                shard_preds[label]['clan_preds'] = clan_preds.cpu().numpy()
                shard_preds[label]['clan_idx'] = clan_idx.cpu().numpy()

    return shard_preds, n_batches

def test_stepROC(data_loader,
               classifier_clan,
               classifier_fam,
               device,
               data_utils,
               hmm_dict,
               pred_dict):

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
                seq_len = len(seqs[idx])
                fam_vector, clan_vector = hu.generate_domain_position_list2(hmm_dict, label, data_utils.maps)
                stop_index = min(seq_len, data_utils.length_limit)
                # Load HMMER* preds
                
                hmm_pred_fam, hmm_pred_clan = hu.generate_domain_position_list3(pred_dict, label, data_utils.maps)
                hmm_pred_fam = torch.tensor(hmm_pred_fam[:stop_index,:])
                hmm_pred_clan = torch.tensor(hmm_pred_clan[:stop_index,:])
                
                fam_vector = np.argmax(fam_vector[:stop_index,:], axis=1) # clip the clan_vector to the truncated sequence length, no need to place on gpu
                clan_vector = np.argmax(clan_vector[:stop_index,:], axis=1) # same as above
                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                clan_preds = classifier_clan(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:])
                clan_preds = F.softmax(clan_preds, dim=1)
                
                ########
                _, preds = classifier_fam(embedding["representations"][data_utils.last_layer][idx,1:stop_index+1,:], clan_preds)

                #########
                ##CLAN_CHUNK_FAM
                clan_fam_weights = data_utils.maps['clan_family_matrix'].to(device)
                # print(torch.nonzero(clan_fam_weights[-1]).squeeze())
                for i in range(clan_fam_weights.shape[0]): #shape is cxf 
                    indices = torch.nonzero(clan_fam_weights[i]).squeeze() #indices for the softmax
                    if i == 656:
                        preds[:,indices] = 1 # IDR is 1:1 map
                    else:
                        preds[:,indices] = F.softmax(preds[:,indices],dim=1)
                
                #########

                #########
                # Multiply by clan
                # expand clan preds
                clan_preds_f = torch.matmul(clan_preds,clan_fam_weights)
                # element wise mul of preds and clan
                preds = preds * clan_preds_f
                #########
                preds = preds.cpu()
                clan_preds = clan_preds.cpu()




                # Store predictions
                shard_preds[label] = {}

                # items to keep for roc plot
                psalm_f_vals, psalm_f_indices = torch.topk(preds, k=1, dim=1)
                psalm_c_vals, psalm_c_indices = torch.topk(clan_preds, k=1, dim=1)
                hmmer_f_vals, hmmer_f_indices = torch.topk(hmm_pred_fam, k=1, dim=1)
                hmmer_c_vals, hmmer_c_indices = torch.topk(hmm_pred_clan, k=1, dim=1)
                
                
                shard_preds[label]['psalm_f_vals'] = psalm_f_vals.numpy()
                shard_preds[label]['psalm_f_indices'] = psalm_f_indices.numpy()
                shard_preds[label]['psalm_c_vals'] = psalm_c_vals.numpy()
                shard_preds[label]['psalm_c_indices'] = psalm_c_indices.numpy()
                shard_preds[label]['hmmer_f_vals'] = hmmer_f_vals.numpy()
                shard_preds[label]['hmmer_f_indices'] = hmmer_f_indices.numpy()
                shard_preds[label]['hmmer_c_vals'] = hmmer_c_vals.numpy()
                shard_preds[label]['hmmer_c_indices'] = hmmer_c_indices.numpy()
                shard_preds[label]["true_f"] = fam_vector
                shard_preds[label]["true_c"] = clan_vector

    return shard_preds, n_batches

def test_stepROC_OH(data_loader,
               classifier_clan,
               classifier_fam,
               device,
               data_utils,
               hmm_dict,
               pred_dict):

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

            sequence_labels = [x.split()[0] for x in labels]

            for idx, label in enumerate(sequence_labels):
                seq_len = len(seqs[idx])
                fam_vector, clan_vector = hu.generate_domain_position_list2(hmm_dict, label, data_utils.maps)
                stop_index = min(seq_len, data_utils.length_limit)
                # Load HMMER* preds
                
                hmm_pred_fam, hmm_pred_clan = hu.generate_domain_position_list3(pred_dict, label, data_utils.maps)
                hmm_pred_fam = torch.tensor(hmm_pred_fam[:stop_index,:])
                hmm_pred_clan = torch.tensor(hmm_pred_clan[:stop_index,:])
                
                fam_vector = np.argmax(fam_vector[:stop_index,:], axis=1) # clip the clan_vector to the truncated sequence length, no need to place on gpu
                clan_vector = np.argmax(clan_vector[:stop_index,:], axis=1) # same as above
                # Logits are the raw output of the classifier!!! This should be used for CrossEntropyLoss()
                # Generate one hot embedding
                embedding = data_utils.get_onehots(seqs[idx][:stop_index]).to(device)
                clan_preds = classifier_clan(embedding)
                clan_preds = F.softmax(clan_preds, dim=1)
                
                ########
                _, preds = classifier_fam(embedding, clan_preds)

                #########
                ##CLAN_CHUNK_FAM
                clan_fam_weights = data_utils.maps['clan_family_matrix'].to(device)
                # print(torch.nonzero(clan_fam_weights[-1]).squeeze())
                for i in range(clan_fam_weights.shape[0]): #shape is cxf 
                    indices = torch.nonzero(clan_fam_weights[i]).squeeze() #indices for the softmax
                    if i == 656:
                        preds[:,indices] = 1 # IDR is 1:1 map
                    else:
                        preds[:,indices] = F.softmax(preds[:,indices],dim=1)
                
                #########

                #########
                # Multiply by clan
                # expand clan preds
                clan_preds_f = torch.matmul(clan_preds,clan_fam_weights)
                # element wise mul of preds and clan
                preds = preds * clan_preds_f
                #########
                preds = preds.cpu()
                clan_preds = clan_preds.cpu()




                # Store predictions
                shard_preds[label] = {}

                # items to keep for roc plot
                psalm_f_vals, psalm_f_indices = torch.topk(preds, k=1, dim=1)
                psalm_c_vals, psalm_c_indices = torch.topk(clan_preds, k=1, dim=1)
                hmmer_f_vals, hmmer_f_indices = torch.topk(hmm_pred_fam, k=1, dim=1)
                hmmer_c_vals, hmmer_c_indices = torch.topk(hmm_pred_clan, k=1, dim=1)
                
                
                shard_preds[label]['psalm_f_vals'] = psalm_f_vals.numpy()
                shard_preds[label]['psalm_f_indices'] = psalm_f_indices.numpy()
                shard_preds[label]['psalm_c_vals'] = psalm_c_vals.numpy()
                shard_preds[label]['psalm_c_indices'] = psalm_c_indices.numpy()
                shard_preds[label]['hmmer_f_vals'] = hmmer_f_vals.numpy()
                shard_preds[label]['hmmer_f_indices'] = hmmer_f_indices.numpy()
                shard_preds[label]['hmmer_c_vals'] = hmmer_c_vals.numpy()
                shard_preds[label]['hmmer_c_indices'] = hmmer_c_indices.numpy()
                shard_preds[label]["true_f"] = fam_vector
                shard_preds[label]["true_c"] = clan_vector

    return shard_preds, n_batches