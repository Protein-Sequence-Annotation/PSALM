import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
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
# Custom LR scheduler for warmup and plateaus
##############################################################

class CustomLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, after_warmup_scheduler):
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.after_warmup_scheduler = after_warmup_scheduler
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(CustomLR, self).__init__(optimizer)

    def get_lr(self):
        if self.step_count < self.warmup_steps:
            warmup_factor = self.step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            self.after_warmup_scheduler.step()  # Update the after warmup scheduler
            return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        self.step_count += 1
        super(CustomLR, self).step(epoch)

##############################################################
# Dataset class to process sequences, create targets and batch
##############################################################

class DistributedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, batches, rank, num_gpus, seed=100):
        self.batches = batches
        self.rank = rank
        self.num_gpus = num_gpus
        self.seed = seed
        if self.seed is not None:
            self.rng = random.Random(self.seed)
        self.max_length_divisible = len(self.batches) - (len(self.batches) % self.num_gpus)
        self.distributed_indices = list(range(self.rank, self.max_length_divisible, self.num_gpus))

    def __iter__(self):

        # self.rng.shuffle(self.distributed_indices) # UNCOMMENT FOR SHUFFLING
        for i in self.distributed_indices:
            yield self.batches[i]
        # # Distribute batches across processes
        # for i in range(self.rank, self.max_length_divisible, self.num_gpus): 
        #     yield self.batches[i]

    def __len__(self):
        return self.max_length_divisible // self.num_gpus

class DataUtils():

    def __init__(self, root, esm_model_name, limit, device):

        with open(Path('info_files') / 'maps.pkl', 'rb') as f:
            self.maps = pickle.load(f)
        self.root = Path(root)
        self.clan_count = len(self.maps["clan_idx"])
        self.fam_count = len(self.maps["fam_idx"])
        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_model.to(device)
        self.last_layer = self.esm_model.num_layers
        self.embedding_dim = self.esm_model.embed_dim
        self.length_limit = limit
        self.tokens_per_batch = 8192 # Can edit as needed
        self.onehot_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

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

        return FastaBatchedDataset.from_file(self.root / f'PSALM_1b_{mode}_{suffix}.fasta')

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

        return self.esm_model(tokens, repr_layers=[self.last_layer], return_contacts=False)["representations"][self.last_layer]

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
    n_batches = len(data_loader)

    classifier.train()

    # if device == 0:
    #     profiler = cProfile.Profile()
    #     profiler.enable()

    for batch_id, (labels, _, tokens) in enumerate(data_loader): # (labels, seqs, tokens)

        # if batch_id % 1000 == 0:
        #     print(f'Rank {device} started batch {batch_id} at {datetime.now()}')
        #     sys.stdout.flush()

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

        # if batch_id == 2000:
        #     return epoch_loss / n_batches

    # if device == 0:
    #     profiler.disable()
    #     stats = pstats.Stats(profiler).sort_stats('ncalls')
    #     stats.dump_stats('check_clan_ml.prof')
        
    return epoch_loss / n_batches

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

        for batch_id, (labels, _, tokens) in enumerate(data_loader):

            tokens = tokens.to(device)
            embedding = data_utils.get_embedding(tokens)

            sequence_labels = [x.split()[0] for x in labels]

            target_vectors = torch.zeros((len(sequence_labels), embedding.shape[1]), dtype=torch.long, device=device)
            mask = torch.zeros((embedding.shape[0], embedding.shape[1]), device=device)

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

            loss = loss_fn(preds[mask.bool()], target_vectors[mask.bool()]) # Average residue loss over all sequences in the batch

            validation_loss += loss.item()

    return validation_loss / n_batches
    
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