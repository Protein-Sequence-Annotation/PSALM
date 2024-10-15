import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn.functional as F
import hmmscan_utils as hu
import pickle
from pathlib import Path
from esm import pretrained, FastaBatchedDataset
import sys
from datetime import datetime
import cProfile, pstats
import random

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
# Custom ReduceLROnPlateau class
##############################################################

from torch.optim.lr_scheduler import ReduceLROnPlateau

class CustomReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=np.sqrt(0.1), patience=3, 
                 threshold=0, threshold_mode='abs', cooldown=2, min_lr=0, eps=1e-8, verbose=False):
        super().__init__(optimizer, mode, factor, patience, threshold, 
                         threshold_mode, cooldown, min_lr, eps, verbose)
        self.prev_metrics = None  # To store previous epoch's metric

    def step(self, metrics, epoch=None):
        current = float(metrics)
        epoch = epoch or self.last_epoch + 1
        self.last_epoch = epoch

        if self.prev_metrics is None:
            self.prev_metrics = current
            return

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # Reset bad epochs during cooldown

        if self._compare(current, self.prev_metrics): #if no improvement wrt prev epoch
            self.num_bad_epochs += 1
        else:
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self.prev_metrics = current

    def _compare(self, current, prev):
        if self.mode == 'min' and current - prev > self.threshold:
            return True
        elif self.mode == 'max' and prev - current > self.threshold:
            return True
        else:
            return False


##############################################################
# Distributed batch sampler for parallelization
##############################################################

class DistributedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, batches, rank, num_gpus, seed=100):
        self.batches = batches
        self.rank = rank
        self.num_gpus = num_gpus
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.max_length_divisible = len(self.batches) - (len(self.batches) % self.num_gpus)
        self.distributed_indices = list(range(self.rank, self.max_length_divisible, self.num_gpus))

    def __iter__(self):

        self.rng.shuffle(self.distributed_indices) # UNCOMMENT FOR SHUFFLING
        for i in self.distributed_indices:
            yield self.batches[i]
        # # Distribute batches across processes
        # for i in range(self.rank, self.max_length_divisible, self.num_gpus): 
        #     yield self.batches[i]

    def __len__(self):
        return self.max_length_divisible // self.num_gpus

##############################################################
# Dataset class to process sequences, create targets and batch
##############################################################

class DataUtils():

    def __init__(self, root, esm_model_name, limit, device, layer_num=None):

        with open(Path('info_files') / 'maps.pkl', 'rb') as f:
            self.maps = pickle.load(f)
        self.root = Path(root)
        self.clan_count = len(self.maps["clan_idx"])
        self.fam_count = len(self.maps["fam_idx"])
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_model.to(device)
        if layer_num is not None and 0 < layer_num <= self.esm_model.num_layers:
            self.extract_layer = layer_num
        else:
            self.extract_layer = self.esm_model.num_layers
        self.embedding_dim = self.esm_model.embed_dim
        self.length_limit = limit
        self.tokens_per_batch = 8192 # Can edit as needed
        self.onehot_alphabet = list(self.alphabet.to_dict().keys())

        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        self.esm_model.eval()
    
    def get_dataset(self, mode, suffix):
        
        """
        Return FastaBatchedDataset from the esm model

        Args:
            idx (int): Shard file index of interest

        Returns:
            dataset (Dataset): dataset with all sequences in shard
        """

        return FastaBatchedDataset.from_file(self.root / f'{mode}{suffix}.fasta')

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
    
    def get_dataloader(self, data, rank, num_gpus):

        """
        Return a data loader for the current dataset

        Args:
            data (Dataset): dataset to pad, batch and crop

        Returns:
            data_loader (DataLoader): a data loader with the appropriate padding function
        """

        seq_lengths = min(max(len(seq) for seq in data.sequence_strs), self.length_limit) # At most 'limit' length
        batches = data.get_batch_indices(self.tokens_per_batch, extra_toks_per_seq=1)

        distributed_batch_sampler = DistributedBatchSampler(batches, rank, num_gpus)

        data_loader = DataLoader(
            data,
            collate_fn=self.alphabet.get_batch_converter(seq_lengths),
            batch_sampler=distributed_batch_sampler,
            pin_memory=True,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )
        
        return data_loader

    def get_embedding(self, tokens):

        """
        Returns the esm embedding for the given sequence

        Args:

        Returns:
            embedding (torch.Tensor): tensor containing the embedding for the given sequence
        """

        return self.esm_model(tokens, repr_layers=[self.extract_layer], return_contacts=False)["representations"][self.extract_layer]

###########################################################
# Train step
###########################################################

def train_step_batch(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict,
               fam):

    """
    Runs a train step for one batch - trains family prediction head

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        loss_fn (nn.Loss): Loss function for the model
        optimizer (torch.optim.Optimizer): Optimizer for the classifier
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): Dictionary with parsed results of hmmscan for current shard
        l1 (bool): Flag to turn on or off the l1 loss
        fam (bool): Flag indicating fam/clan model

    Returns:
        epoch_loss (torch.Float32): Average residue loss over the entire epoch
    """

    epoch_loss = torch.tensor(0., device=device)
    correct_predictions = torch.tensor(0., device=device)
    total_predictions = torch.tensor(0., device=device)
    TP, FP, TN, FN = torch.tensor(0., device=device), torch.tensor(0., device=device),\
                    torch.tensor(0., device=device), torch.tensor(0., device=device)
    
    n_batches = len(data_loader)

    classifier.train()
    
    for batch_id, (labels, _, tokens) in enumerate(data_loader): # (labels, seqs, tokens)

        tokens = tokens.to(device)
        embedding = data_utils.get_embedding(tokens)

        optimizer.zero_grad()

        sequence_labels = [x.split()[0] for x in labels]

        target_vectors = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)
        mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)
        
        if fam:
            clan_support = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)

        for idx, label in enumerate(sequence_labels):

            fam_vector = hmm_dict[label]['fam_vector']
            clan_vector = hmm_dict[label]['clan_vector']
            stop_index = hmm_dict[label]['stop']
            
            if fam:
                clan_support[idx, 1:stop_index+1] = clan_vector
                target_vectors[idx, 1:stop_index+1] = fam_vector
            else:
                target_vectors[idx, 1:stop_index+1] = clan_vector

            mask[idx, 1:stop_index+1] = True

        if fam:
            clan_support =  F.one_hot(clan_support, data_utils.clan_count).float()
            preds, _ = classifier(embedding, mask, clan_support)
        else:
            preds = classifier(embedding, mask)

        # Average residue loss over all sequences in the batch

        loss = loss_fn(preds[mask], target_vectors[mask])
        
        loss.backward()
        optimizer.step()

        # epoch_loss += loss.item()
        epoch_loss += loss.detach()

        # Calculate correct predictions (only for masked positions)
        # Accuracy calculation
        predicted_labels = preds.argmax(dim=-1)  # Assuming your task involves selecting the highest probability label
        correct_predictions += (predicted_labels == target_vectors).masked_select(mask).sum().item()
        total_predictions += mask.sum().item()
        
        # Calculate TPR/FPR
        predicted_masked = predicted_labels[mask]
        target_masked = target_vectors[mask]
        is_IDR = 19632 if fam else 656
        not_IDR = predicted_masked != is_IDR
        matches = predicted_masked == target_masked
        predicted_is_IDR = predicted_masked == is_IDR
        target_is_IDR = target_masked == is_IDR
        # True Positives: Matches and neither is IDR
        TP += (matches & not_IDR & (target_masked != is_IDR)).sum().item()
        # True Negatives: Both are IDR
        TN += (predicted_is_IDR & target_is_IDR).sum().item()
        # False Positives: Target is IDR but predicted is not
        FP += (~matches & target_is_IDR & not_IDR).sum().item()
        # False Negatives: Predicted is IDR but target is not
        FN += (~matches & predicted_is_IDR & (target_masked != is_IDR)).sum().item()
        
    return epoch_loss / n_batches, correct_predictions/total_predictions, TP/(TP+FN), FP/(FP+TN)

def train_step_batch_onehot(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict,
               fam):

    """
    Runs a train step for one batch - trains family prediction head for onehot baseline models

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        loss_fn (nn.Loss): Loss function for the model
        optimizer (torch.optim.Optimizer): Optimizer for the classifier
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): Dictionary with parsed results of hmmscan for current shard
        l1 (bool): Flag to turn on or off the l1 loss
        fam (bool): Flag indicating fam/clan model

    Returns:
        epoch_loss (torch.Float32): Average residue loss over the entire epoch
    """

    epoch_loss = torch.tensor(0., device=device)
    correct_predictions = torch.tensor(0., device=device)
    total_predictions = torch.tensor(0., device=device)
    TP, FP, TN, FN = torch.tensor(0., device=device), torch.tensor(0., device=device),\
                    torch.tensor(0., device=device), torch.tensor(0., device=device)
    
    n_batches = len(data_loader)

    classifier.train()
    
    for batch_id, (labels, _, tokens) in enumerate(data_loader): # (labels, seqs, tokens)

        tokens = tokens.to(device)

        # get onehot
        alphabet_size = len(data_utils.onehot_alphabet)
        embedding = F.one_hot(tokens, num_classes = alphabet_size).float()

        optimizer.zero_grad()

        sequence_labels = [x.split()[0] for x in labels]

        target_vectors = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)
        mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)
        
        if fam:
            clan_support = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)

        for idx, label in enumerate(sequence_labels):

            fam_vector = hmm_dict[label]['fam_vector']
            clan_vector = hmm_dict[label]['clan_vector']
            stop_index = hmm_dict[label]['stop']
            
            if fam:
                clan_support[idx, 1:stop_index+1] = clan_vector
                target_vectors[idx, 1:stop_index+1] = fam_vector
            else:
                target_vectors[idx, 1:stop_index+1] = clan_vector

            mask[idx, 1:stop_index+1] = True
            # mask the onehot embedding appropriately
            

        if fam:
            clan_support =  F.one_hot(clan_support, data_utils.clan_count).float()
            preds, _ = classifier(embedding, mask, clan_support)
        else:
            preds = classifier(embedding, mask)

        # Average residue loss over all sequences in the batch

        loss = loss_fn(preds[mask], target_vectors[mask])

        loss.backward()
        optimizer.step()

        # epoch_loss += loss.item()
        epoch_loss += loss.detach()

        # Calculate correct predictions (only for masked positions)
        # Accuracy calculation
        predicted_labels = preds.argmax(dim=-1)  # Assuming your task involves selecting the highest probability label
        correct_predictions += (predicted_labels == target_vectors).masked_select(mask).sum().item()
        total_predictions += mask.sum().item()
        
        # Calculate TPR/FPR
        predicted_masked = predicted_labels[mask]
        target_masked = target_vectors[mask]
        is_IDR = 19632 if fam else 656
        not_IDR = predicted_masked != is_IDR
        matches = predicted_masked == target_masked
        predicted_is_IDR = predicted_masked == is_IDR
        target_is_IDR = target_masked == is_IDR
        # True Positives: Matches and neither is IDR
        TP += (matches & not_IDR & (target_masked != is_IDR)).sum().item()
        # True Negatives: Both are IDR
        TN += (predicted_is_IDR & target_is_IDR).sum().item()
        # False Positives: Target is IDR but predicted is not
        FP += (~matches & target_is_IDR & not_IDR).sum().item()
        # False Negatives: Predicted is IDR but target is not
        FN += (~matches & predicted_is_IDR & (target_masked != is_IDR)).sum().item()

    return epoch_loss / n_batches, correct_predictions/total_predictions, TP/(TP+FN), FP/(FP+TN)

def train_step_batch_onlyfams(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict,
               fam):

    """
    Runs a train step for one batch - trains family prediction head

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        loss_fn (nn.Loss): Loss function for the model
        optimizer (torch.optim.Optimizer): Optimizer for the classifier
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): Dictionary with parsed results of hmmscan for current shard
        l1 (bool): Flag to turn on or off the l1 loss
        fam (bool): Flag indicating fam/clan model

    Returns:
        epoch_loss (torch.Float32): Average residue loss over the entire epoch
    """

    epoch_loss = torch.tensor(0., device=device)
    correct_predictions = torch.tensor(0., device=device)
    total_predictions = torch.tensor(0., device=device)
    TP, FP, TN, FN = torch.tensor(0., device=device), torch.tensor(0., device=device),\
                    torch.tensor(0., device=device), torch.tensor(0., device=device)
    
    n_batches = len(data_loader)

    classifier.train()
    
    for batch_id, (labels, _, tokens) in enumerate(data_loader): # (labels, seqs, tokens)

        tokens = tokens.to(device)
        embedding = data_utils.get_embedding(tokens)

        optimizer.zero_grad()

        sequence_labels = [x.split()[0] for x in labels]

        target_vectors = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)
        mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)
        
        for idx, label in enumerate(sequence_labels):

            fam_vector = hmm_dict[label]['fam_vector']
            stop_index = hmm_dict[label]['stop']
            
            target_vectors[idx, 1:stop_index+1] = fam_vector

            mask[idx, 1:stop_index+1] = True

        preds = classifier(embedding, mask)

        # Average residue loss over all sequences in the batch

        loss = loss_fn(preds[mask], target_vectors[mask])
        
        loss.backward()
        optimizer.step()

        # epoch_loss += loss.item()
        epoch_loss += loss.detach()

        # Calculate correct predictions (only for masked positions)
        # Accuracy calculation
        predicted_labels = preds.argmax(dim=-1)  # Assuming your task involves selecting the highest probability label
        correct_predictions += (predicted_labels == target_vectors).masked_select(mask).sum().item()
        total_predictions += mask.sum().item()
        
        # Calculate TPR/FPR
        predicted_masked = predicted_labels[mask]
        target_masked = target_vectors[mask]
        is_IDR = 19632
        not_IDR = predicted_masked != is_IDR
        matches = predicted_masked == target_masked
        predicted_is_IDR = predicted_masked == is_IDR
        target_is_IDR = target_masked == is_IDR
        # True Positives: Matches and neither is IDR
        TP += (matches & not_IDR & (target_masked != is_IDR)).sum().item()
        # True Negatives: Both are IDR
        TN += (predicted_is_IDR & target_is_IDR).sum().item()
        # False Positives: Target is IDR but predicted is not
        FP += (~matches & target_is_IDR & not_IDR).sum().item()
        # False Negatives: Predicted is IDR but target is not
        FN += (~matches & predicted_is_IDR & (target_masked != is_IDR)).sum().item()
        
    return epoch_loss / n_batches, correct_predictions/total_predictions, TP/(TP+FN), FP/(FP+TN)

def train_step_batch_onlyfams_onehot(data_loader,
               classifier,
               loss_fn,
               optimizer,
               device,
               data_utils,
               hmm_dict,
               fam):

    """
    Runs a train step for one batch - trains family prediction head

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        loss_fn (nn.Loss): Loss function for the model
        optimizer (torch.optim.Optimizer): Optimizer for the classifier
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): Dictionary with parsed results of hmmscan for current shard
        l1 (bool): Flag to turn on or off the l1 loss
        fam (bool): Flag indicating fam/clan model

    Returns:
        epoch_loss (torch.Float32): Average residue loss over the entire epoch
    """

    epoch_loss = torch.tensor(0., device=device)
    correct_predictions = torch.tensor(0., device=device)
    total_predictions = torch.tensor(0., device=device)
    TP, FP, TN, FN = torch.tensor(0., device=device), torch.tensor(0., device=device),\
                    torch.tensor(0., device=device), torch.tensor(0., device=device)
    
    n_batches = len(data_loader)

    classifier.train()
    
    for batch_id, (labels, _, tokens) in enumerate(data_loader): # (labels, seqs, tokens)

        tokens = tokens.to(device)
        alphabet_size = len(data_utils.onehot_alphabet)
        embedding = F.one_hot(tokens, num_classes = alphabet_size).float()

        optimizer.zero_grad()

        sequence_labels = [x.split()[0] for x in labels]

        target_vectors = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)
        mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)
        
        for idx, label in enumerate(sequence_labels):

            fam_vector = hmm_dict[label]['fam_vector']
            stop_index = hmm_dict[label]['stop']
            
            target_vectors[idx, 1:stop_index+1] = fam_vector

            mask[idx, 1:stop_index+1] = True

        preds = classifier(embedding, mask)

        # Average residue loss over all sequences in the batch

        loss = loss_fn(preds[mask], target_vectors[mask])
        
        loss.backward()
        optimizer.step()

        # epoch_loss += loss.item()
        epoch_loss += loss.detach()

        # Calculate correct predictions (only for masked positions)
        # Accuracy calculation
        predicted_labels = preds.argmax(dim=-1)  # Assuming your task involves selecting the highest probability label
        correct_predictions += (predicted_labels == target_vectors).masked_select(mask).sum().item()
        total_predictions += mask.sum().item()
        
        # Calculate TPR/FPR
        predicted_masked = predicted_labels[mask]
        target_masked = target_vectors[mask]
        is_IDR = 19632
        not_IDR = predicted_masked != is_IDR
        matches = predicted_masked == target_masked
        predicted_is_IDR = predicted_masked == is_IDR
        target_is_IDR = target_masked == is_IDR
        # True Positives: Matches and neither is IDR
        TP += (matches & not_IDR & (target_masked != is_IDR)).sum().item()
        # True Negatives: Both are IDR
        TN += (predicted_is_IDR & target_is_IDR).sum().item()
        # False Positives: Target is IDR but predicted is not
        FP += (~matches & target_is_IDR & not_IDR).sum().item()
        # False Negatives: Predicted is IDR but target is not
        FN += (~matches & predicted_is_IDR & (target_masked != is_IDR)).sum().item()
        
    return epoch_loss / n_batches, correct_predictions/total_predictions, TP/(TP+FN), FP/(FP+TN)

###########################################################
# Validation step
###########################################################

def validate_batch(data_loader,
               classifier,
               loss_fn,
               device,
               data_utils,
               hmm_dict,
               fam):

    """
    Runs a validation step for one batch - only clan prediction head

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): Dictionary with parsed results of hmmscan for current shard
        fam (bool): Flag indicating family/clan model

    Returns:
        validation_loss (float): Loss value for validation set
    """

    n_batches = len(data_loader)
    classifier.eval()

    with torch.inference_mode():

        validation_loss = torch.tensor(0., device=device)
        correct_predictions = torch.tensor(0., device=device)
        total_predictions = torch.tensor(0., device=device)
        TP, FP, TN, FN = torch.tensor(0., device=device), torch.tensor(0., device=device),\
                        torch.tensor(0., device=device), torch.tensor(0., device=device)

        for batch_id, (labels, _, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]

            target_vectors = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)
            mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)

            if fam:
                clan_support = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)

            for idx, label in enumerate(sequence_labels):

                fam_vector = hmm_dict[label]['fam_vector']
                clan_vector = hmm_dict[label]['clan_vector']
                stop_index = hmm_dict[label]['stop']

                if fam:
                    target = fam_vector
                    clan_support[idx, 1:stop_index+1] = clan_vector
                else:
                    target = clan_vector
            
                target_vectors[idx, 1:stop_index+1] = target
                mask[idx, 1:stop_index+1] = 1
            
            if fam:
                clan_support = F.one_hot(clan_support, data_utils.clan_count).float()
                preds, _ = classifier(embedding, mask, clan_support)
            else:
                preds = classifier(embedding, mask)

            loss = loss_fn(preds[mask], target_vectors[mask]) # Average residue loss over all sequences in the batch

            validation_loss += loss.item()

            # Accuracy calculation
            predicted_labels = preds.argmax(dim=-1)  # Assuming your task involves selecting the highest probability label

            # Calculate correct predictions (only for masked positions)
            correct_predictions += (predicted_labels == target_vectors).masked_select(mask.bool()).sum().item()
            total_predictions += mask.sum().item()

            # Calculate TPR/FPR
            predicted_masked = predicted_labels[mask]
            target_masked = target_vectors[mask]
            is_IDR = 19632 if fam else 656
            not_IDR = predicted_masked != is_IDR
            matches = predicted_masked == target_masked
            predicted_is_IDR = predicted_masked == is_IDR
            target_is_IDR = target_masked == is_IDR
            # True Positives: Matches and neither is IDR
            TP += (matches & not_IDR & (target_masked != is_IDR)).sum().item()
            # True Negatives: Both are IDR
            TN += (predicted_is_IDR & target_is_IDR).sum().item()
            # False Positives: Target is IDR but predicted is not
            FP += (~matches & target_is_IDR & not_IDR).sum().item()
            # False Negatives: Predicted is IDR but target is not
            FN += (~matches & predicted_is_IDR & (target_masked != is_IDR)).sum().item()

    return validation_loss / n_batches, correct_predictions/total_predictions, TP/(TP+FN), FP/(FP+TN)

def validate_batch_onehot(data_loader,
               classifier,
               loss_fn,
               device,
               data_utils,
               hmm_dict,
               fam):

    """
    Runs a validation step for one batch - only clan prediction head

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): Dictionary with parsed results of hmmscan for current shard
        fam (bool): Flag indicating family/clan model

    Returns:
        validation_loss (float): Loss value for validation set
    """

    n_batches = len(data_loader)
    classifier.eval()

    with torch.inference_mode():

        validation_loss = torch.tensor(0., device=device)
        correct_predictions = torch.tensor(0., device=device)
        total_predictions = torch.tensor(0., device=device)
        TP, FP, TN, FN = torch.tensor(0., device=device), torch.tensor(0., device=device),\
                        torch.tensor(0., device=device), torch.tensor(0., device=device)

        for batch_id, (labels, _, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            # get onehot
            alphabet_size = len(data_utils.onehot_alphabet)
            embedding = F.one_hot(tokens, alphabet_size).float()

            sequence_labels = [x.split()[0] for x in labels]

            target_vectors = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)
            mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)

            if fam:
                clan_support = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)

            for idx, label in enumerate(sequence_labels):

                fam_vector = hmm_dict[label]['fam_vector']
                clan_vector = hmm_dict[label]['clan_vector']
                stop_index = hmm_dict[label]['stop']

                if fam:
                    target = fam_vector
                    clan_support[idx, 1:stop_index+1] = clan_vector
                else:
                    target = clan_vector
            
                target_vectors[idx, 1:stop_index+1] = target
                mask[idx, 1:stop_index+1] = 1
            
            if fam:
                clan_support = F.one_hot(clan_support, data_utils.clan_count).float()
                preds, _ = classifier(embedding, mask, clan_support)
            else:
                preds = classifier(embedding, mask)

            loss = loss_fn(preds[mask], target_vectors[mask]) # Average residue loss over all sequences in the batch

            validation_loss += loss.item()

            # Accuracy calculation
            predicted_labels = preds.argmax(dim=-1)  # Assuming your task involves selecting the highest probability label

            # Calculate correct predictions (only for masked positions)
            correct_predictions += (predicted_labels == target_vectors).masked_select(mask.bool()).sum().item()
            total_predictions += mask.sum().item()

            # Calculate TPR/FPR
            predicted_masked = predicted_labels[mask]
            target_masked = target_vectors[mask]
            is_IDR = 19632 if fam else 656
            not_IDR = predicted_masked != is_IDR
            matches = predicted_masked == target_masked
            predicted_is_IDR = predicted_masked == is_IDR
            target_is_IDR = target_masked == is_IDR
            # True Positives: Matches and neither is IDR
            TP += (matches & not_IDR & (target_masked != is_IDR)).sum().item()
            # True Negatives: Both are IDR
            TN += (predicted_is_IDR & target_is_IDR).sum().item()
            # False Positives: Target is IDR but predicted is not
            FP += (~matches & target_is_IDR & not_IDR).sum().item()
            # False Negatives: Predicted is IDR but target is not
            FN += (~matches & predicted_is_IDR & (target_masked != is_IDR)).sum().item()

    return validation_loss / n_batches, correct_predictions/total_predictions, TP/(TP+FN), FP/(FP+TN)

def validate_batch_onlyfams(data_loader,
               classifier,
               loss_fn,
               device,
               data_utils,
               hmm_dict,
               fam):

    """
    Runs a validation step for one batch - only clan prediction head

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): Dictionary with parsed results of hmmscan for current shard
        fam (bool): Flag indicating family/clan model

    Returns:
        validation_loss (float): Loss value for validation set
    """

    n_batches = len(data_loader)
    classifier.eval()

    with torch.inference_mode():

        validation_loss = torch.tensor(0., device=device)
        correct_predictions = torch.tensor(0., device=device)
        total_predictions = torch.tensor(0., device=device)
        TP, FP, TN, FN = torch.tensor(0., device=device), torch.tensor(0., device=device),\
                        torch.tensor(0., device=device), torch.tensor(0., device=device)

        for batch_id, (labels, _, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]

            target_vectors = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)
            mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)

            for idx, label in enumerate(sequence_labels):

                fam_vector = hmm_dict[label]['fam_vector']
                stop_index = hmm_dict[label]['stop']

                target = fam_vector
            
                target_vectors[idx, 1:stop_index+1] = target
                mask[idx, 1:stop_index+1] = 1
            
            preds = classifier(embedding, mask)

            loss = loss_fn(preds[mask], target_vectors[mask]) # Average residue loss over all sequences in the batch

            validation_loss += loss.item()

            # Accuracy calculation
            predicted_labels = preds.argmax(dim=-1)  # Assuming your task involves selecting the highest probability label

            # Calculate correct predictions (only for masked positions)
            correct_predictions += (predicted_labels == target_vectors).masked_select(mask.bool()).sum().item()
            total_predictions += mask.sum().item()

            # Calculate TPR/FPR
            predicted_masked = predicted_labels[mask]
            target_masked = target_vectors[mask]
            is_IDR = 19632
            not_IDR = predicted_masked != is_IDR
            matches = predicted_masked == target_masked
            predicted_is_IDR = predicted_masked == is_IDR
            target_is_IDR = target_masked == is_IDR
            # True Positives: Matches and neither is IDR
            TP += (matches & not_IDR & (target_masked != is_IDR)).sum().item()
            # True Negatives: Both are IDR
            TN += (predicted_is_IDR & target_is_IDR).sum().item()
            # False Positives: Target is IDR but predicted is not
            FP += (~matches & target_is_IDR & not_IDR).sum().item()
            # False Negatives: Predicted is IDR but target is not
            FN += (~matches & predicted_is_IDR & (target_masked != is_IDR)).sum().item()

    return validation_loss / n_batches, correct_predictions/total_predictions, TP/(TP+FN), FP/(FP+TN)

def validate_batch_onlyfams_onehot(data_loader,
               classifier,
               loss_fn,
               device,
               data_utils,
               hmm_dict,
               fam):

    """
    Runs a validation step for one batch - only clan prediction head

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        classifier (nn.Module): The classifier head to decode esm embeddings
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): Dictionary with parsed results of hmmscan for current shard
        fam (bool): Flag indicating family/clan model

    Returns:
        validation_loss (float): Loss value for validation set
    """

    n_batches = len(data_loader)
    classifier.eval()

    with torch.inference_mode():

        validation_loss = torch.tensor(0., device=device)
        correct_predictions = torch.tensor(0., device=device)
        total_predictions = torch.tensor(0., device=device)
        TP, FP, TN, FN = torch.tensor(0., device=device), torch.tensor(0., device=device),\
                        torch.tensor(0., device=device), torch.tensor(0., device=device)

        for batch_id, (labels, _, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            alphabet_size = len(data_utils.onehot_alphabet)
            embedding = F.one_hot(tokens, alphabet_size).float()

            sequence_labels = [x.split()[0] for x in labels]

            target_vectors = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)
            mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)

            for idx, label in enumerate(sequence_labels):

                fam_vector = hmm_dict[label]['fam_vector']
                stop_index = hmm_dict[label]['stop']

                target = fam_vector
            
                target_vectors[idx, 1:stop_index+1] = target
                mask[idx, 1:stop_index+1] = 1
            
            preds = classifier(embedding, mask)

            loss = loss_fn(preds[mask], target_vectors[mask]) # Average residue loss over all sequences in the batch

            validation_loss += loss.item()

            # Accuracy calculation
            predicted_labels = preds.argmax(dim=-1)  # Assuming your task involves selecting the highest probability label

            # Calculate correct predictions (only for masked positions)
            correct_predictions += (predicted_labels == target_vectors).masked_select(mask.bool()).sum().item()
            total_predictions += mask.sum().item()

            # Calculate TPR/FPR
            predicted_masked = predicted_labels[mask]
            target_masked = target_vectors[mask]
            is_IDR = 19632
            not_IDR = predicted_masked != is_IDR
            matches = predicted_masked == target_masked
            predicted_is_IDR = predicted_masked == is_IDR
            target_is_IDR = target_masked == is_IDR
            # True Positives: Matches and neither is IDR
            TP += (matches & not_IDR & (target_masked != is_IDR)).sum().item()
            # True Negatives: Both are IDR
            TN += (predicted_is_IDR & target_is_IDR).sum().item()
            # False Positives: Target is IDR but predicted is not
            FP += (~matches & target_is_IDR & not_IDR).sum().item()
            # False Negatives: Predicted is IDR but target is not
            FN += (~matches & predicted_is_IDR & (target_masked != is_IDR)).sum().item()

    return validation_loss / n_batches, correct_predictions/total_predictions, TP/(TP+FN), FP/(FP+TN)

###########################################################
# Test step
###########################################################

def test_step_batch(data_loader,
               clan_classifier,
               fam_classifier,
               device,
               data_utils,
               hmm_dict,
               fam,
               save_path):

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

    gpu_preds = {}

    n_batches = len(data_loader)
    clan_classifier.eval()
    fam_classifier.eval()

    with torch.inference_mode():

        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]
            stop_indices = [min(len(seq), data_utils.length_limit) for seq in seqs]

            mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)

            for idx, label in enumerate(sequence_labels):

                mask[idx, 1:stop_indices[idx]+1] = 1
            
            clan_preds = clan_classifier(embedding, mask)
            clan_preds = F.softmax(clan_preds, dim=2) ###
            fam_preds,_ = fam_classifier(embedding, mask, clan_preds)

            # Clan based normalization
            clan_fam_weights = data_utils.maps['clan_family_matrix'].to(device)
            for i in range(clan_fam_weights.shape[0]):  # shape is cxf 
                indices = torch.nonzero(clan_fam_weights[i]).squeeze()  # indices for the softmax
                if i == 656:
                    fam_preds[:, :, indices] = 1  # IDR is 1:1 map
                else:
                    fam_preds[:, :, indices] = torch.softmax(fam_preds[:, :, indices], dim=2)     

            # Multiply by clan, expand clan preds
            clan_preds_f = torch.matmul(clan_preds, clan_fam_weights)
            fam_preds = fam_preds * clan_preds_f

            clan_preds, clan_idx = torch.topk(clan_preds, k=1, dim=2)
            fam_preds, fam_idx = torch.topk(fam_preds, k=1, dim=2)

            clan_preds = clan_preds.cpu().numpy()
            clan_idx = clan_idx.cpu().numpy()
            fam_preds = fam_preds.cpu().numpy()
            fam_idx = fam_idx.cpu().numpy()

            # Store predictions

            for label_idx in range(len(stop_indices)):
                
                label = sequence_labels[label_idx]
                gpu_preds[label] = {}

                gpu_preds[label]['clan_preds'] = clan_preds[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['clan_idx'] = clan_idx[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['clan_true'] = hmm_dict[label]['clan_vector'].numpy()
                gpu_preds[label]['fam_preds'] = fam_preds[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['fam_idx'] = fam_idx[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['fam_true'] = hmm_dict[label]['fam_vector'].numpy()

    with open(f"{save_path}/gpu_{device}_preds.pkl", 'wb') as f:
        pickle.dump(gpu_preds, f)

def test_step_batch_onehot(data_loader,
                           clan_classifier,
                           fam_classifier,
                           device,
                           data_utils,
                           hmm_dict,
                           fam,
                           save_path):
    """
    Runs a test step for one batch - only clan prediction head with onehot inputs.

    Args:
        data_loader (DataLoader): A data loader object with the current dataset
        clan_classifier (nn.Module): Clan classifier head
        fam_classifier (nn.Module): Family classifier head
        device (str): GPU / CPU selection
        data_utils (DataUtils): Member functions and helpers for processing HMM data
        hmm_dict (Dict): Dictionary with parsed results of hmmscan for the current shard
        fam (bool): Flag indicating family/clan model
        save_path (str): Path to save predictions

    Returns:
        gpu_preds (Dict): Predictions for each sequence: top 2 values, indices, target
    """

    gpu_preds = {}
    n_batches = len(data_loader)
    clan_classifier.eval()
    fam_classifier.eval()

    with torch.inference_mode():
        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):
            tokens = tokens.to(device)

            # Convert tokens to one-hot encoded embeddings
            alphabet_size = len(data_utils.onehot_alphabet)
            embedding = F.one_hot(tokens, alphabet_size).float()

            sequence_labels = [x.split()[0] for x in labels]
            stop_indices = [min(len(seq), data_utils.length_limit) for seq in seqs]

            mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)

            for idx, label in enumerate(sequence_labels):
                mask[idx, 1:stop_indices[idx]+1] = 1
            
            # Clan classifier predictions using one-hot encoded embeddings
            clan_preds = clan_classifier(embedding, mask)
            clan_preds = F.softmax(clan_preds, dim=2)  # Softmax over clan predictions
            
            # Family classifier predictions using clan predictions
            fam_preds, _ = fam_classifier(embedding, mask, clan_preds)

            # Clan-based normalization
            clan_fam_weights = data_utils.maps['clan_family_matrix'].to(device)
            for i in range(clan_fam_weights.shape[0]):  # shape is cxf 
                indices = torch.nonzero(clan_fam_weights[i]).squeeze()  # indices for softmax
                if i == 656:
                    fam_preds[:, :, indices] = 1  # IDR is 1:1 map
                else:
                    fam_preds[:, :, indices] = torch.softmax(fam_preds[:, :, indices], dim=2)

            # Multiply family predictions by clan predictions, expand clan preds
            clan_preds_f = torch.matmul(clan_preds, clan_fam_weights)
            fam_preds = fam_preds * clan_preds_f

            # Get top predictions for clans and families
            clan_preds, clan_idx = torch.topk(clan_preds, k=1, dim=2)
            fam_preds, fam_idx = torch.topk(fam_preds, k=1, dim=2)

            # Move predictions to CPU and convert to numpy for storage
            clan_preds = clan_preds.cpu().numpy()
            clan_idx = clan_idx.cpu().numpy()
            fam_preds = fam_preds.cpu().numpy()
            fam_idx = fam_idx.cpu().numpy()

            # Store predictions in gpu_preds dictionary
            for label_idx in range(len(stop_indices)):
                label = sequence_labels[label_idx]
                gpu_preds[label] = {}

                gpu_preds[label]['clan_preds'] = clan_preds[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['clan_idx'] = clan_idx[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['clan_true'] = hmm_dict[label]['clan_vector'].numpy()
                gpu_preds[label]['fam_preds'] = fam_preds[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['fam_idx'] = fam_idx[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['fam_true'] = hmm_dict[label]['fam_vector'].numpy()

    # Save the predictions to the specified path
    with open(f"{save_path}/gpu_{device}_preds.pkl", 'wb') as f:
        pickle.dump(gpu_preds, f)

    return gpu_preds

def test_step_batch_onlyfams(data_loader,
               fam_classifier,
               device,
               data_utils,
               hmm_dict,
               fam,
               save_path):

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

    gpu_preds = {}

    n_batches = len(data_loader)
    fam_classifier.eval()

    with torch.inference_mode():

        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]
            stop_indices = [min(len(seq), data_utils.length_limit) for seq in seqs]

            mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)

            for idx, label in enumerate(sequence_labels):

                mask[idx, 1:stop_indices[idx]+1] = 1
            
            fam_preds = fam_classifier(embedding, mask)

            fam_preds, fam_idx = torch.topk(fam_preds, k=1, dim=2)

            fam_preds = fam_preds.cpu().numpy()
            fam_idx = fam_idx.cpu().numpy()

            # Store predictions

            for label_idx in range(len(stop_indices)):
                
                label = sequence_labels[label_idx]
                gpu_preds[label] = {}
                tmp_holder = fam_idx[label_idx, 1:stop_indices[label_idx]+1].ravel()
                
                gpu_preds[label]['clan_preds'] = fam_preds[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['clan_idx'] = np.array([data_utils.maps['clan_idx'][data_utils.maps['fam_clan'][data_utils.maps['idx_fam'][entry]]] for entry in tmp_holder])
                gpu_preds[label]['clan_true'] = hmm_dict[label]['clan_vector'].numpy()
                gpu_preds[label]['fam_preds'] = fam_preds[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['fam_idx'] = fam_idx[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['fam_true'] = hmm_dict[label]['fam_vector'].numpy()

    with open(f"{save_path}/gpu_{device}_preds.pkl", 'wb') as f:
        pickle.dump(gpu_preds, f)

def test_step_batch_onlyfams_onehot(data_loader,
               fam_classifier,
               device,
               data_utils,
               hmm_dict,
               fam,
               save_path):

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

    gpu_preds = {}

    n_batches = len(data_loader)
    fam_classifier.eval()

    with torch.inference_mode():

        for batch_id, (labels, seqs, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            alphabet_size = len(data_utils.onehot_alphabet)
            embedding = F.one_hot(tokens, alphabet_size).float()

            sequence_labels = [x.split()[0] for x in labels]
            stop_indices = [min(len(seq), data_utils.length_limit) for seq in seqs]

            mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device, dtype=torch.bool)

            for idx, label in enumerate(sequence_labels):

                mask[idx, 1:stop_indices[idx]+1] = 1
            
            fam_preds = fam_classifier(embedding, mask)

            fam_preds, fam_idx = torch.topk(fam_preds, k=1, dim=2)

            fam_preds = fam_preds.cpu().numpy()
            fam_idx = fam_idx.cpu().numpy()

            # Store predictions

            for label_idx in range(len(stop_indices)):
                
                label = sequence_labels[label_idx]
                gpu_preds[label] = {}
                tmp_holder = fam_idx[label_idx, 1:stop_indices[label_idx]+1].ravel()
                
                gpu_preds[label]['clan_preds'] = fam_preds[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['clan_idx'] = np.array([data_utils.maps['clan_idx'][data_utils.maps['fam_clan'][data_utils.maps['idx_fam'][entry]]] for entry in tmp_holder])
                gpu_preds[label]['clan_true'] = hmm_dict[label]['clan_vector'].numpy()
                gpu_preds[label]['fam_preds'] = fam_preds[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['fam_idx'] = fam_idx[label_idx, 1:stop_indices[label_idx]+1]
                gpu_preds[label]['fam_true'] = hmm_dict[label]['fam_vector'].numpy()

    with open(f"{save_path}/gpu_{device}_preds.pkl", 'wb') as f:
        pickle.dump(gpu_preds, f)