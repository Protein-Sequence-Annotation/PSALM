import ml_utils as mu
import classifiers as cf
import torch
import sys
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
import os, sys
import torch.distributed as dist
from datetime import timedelta, datetime
import cProfile, pstats
import pickle
import numpy as np

def setup(rank, num_gpus):

    """
    Setting up process group for distributed training

    Args:
        rank (int) - rank of current process
        num_gpus (int) - number of GPUs
    """

    os.environ['MASTER_ADDR'] = '10.31.179.219' # IP address of GPU node
    os.environ['MASTER_PORT'] = '12355' # Free port number

    dist.init_process_group("nccl", rank=rank, world_size=num_gpus, timeout=timedelta(minutes=30))

def cleanup():

    """
    Clear distributed model and free memory after training
    """

    dist.destroy_process_group()

def loadAll(rank, num_gpus, args, save_path, test_dict_path):

    """
    Function to run on all processes in distributed training

    Args:
        rank (int): Rank of current process
        num_gpus (int) - number of GPUs
        args (ArgParser): set of parsed commandline arguments
        save_path (Path): Location to save checkpoint
        train_dict (Dict): Dictionary of all HMM scans for train set
        val_dict (Dict): Dictionary of all HMM scans for validation set    
    """
    
    setup(rank, num_gpus) # Setup process group
    torch.set_float32_matmul_precision('high') # Lower precision computing
    torch.backends.cuda.matmul.allow_tf32 = True # Use TF32 cores
    torch.backends.cudnn.allow_tf32 = True # Use TF32 cores

    # Load scans for train and validation
    with open(test_dict_path, 'rb') as f:
        test_dict = pickle.load(f)

    # Instantiate dataset helper object
    
    length_limit = 4096 # Covers 99.75% sequences
    esm_model_name =  'esm2_t33_650M_UR50D' # ESM 650 million parameter model

    data_utils = mu.DataUtils(args.root, esm_model_name, length_limit, rank, layer_num = args.layer_number) # DataUtils object
    
    # Model choice based on user input

    clan_classifier = cf.ClanLSTMbatch(data_utils.embedding_dim,data_utils.clan_count).to(rank)
    clan_classifier.load_state_dict(torch.load(args.clan_filename))

    fam_classifier = cf.FamLSTMbatch(data_utils.embedding_dim, data_utils.maps, rank).to(rank)
    fam_classifier.load_state_dict(torch.load(args.fam_filename))

    clan_classifier = torch.compile(clan_classifier, mode='max-autotune') # Pre compile for optimization
    fam_classifier = torch.compile(fam_classifier, mode='max-autotune') # Pre compile for optimization

    clan_classifier = DDP(clan_classifier, device_ids=[rank], output_device=rank, find_unused_parameters=False) # DDP
    fam_classifier = DDP(clan_classifier, device_ids=[rank], output_device=rank, find_unused_parameters=False) # DDP

    # Load validation set if master process - will change once parallelized
    
    # Parameters for training loop
    dataset = data_utils.get_dataset(mode='Test', suffix=args.suffix) # Get corresponding dataset from fasta
    data_loader = data_utils.get_dataloader(dataset, rank, num_gpus) # Parallelize data loader

    num_epochs = args.num_epochs

    # Initialize wandb if master process

    if not args.no_log and rank == 0:
        run = wandb.init(project=args.project, 
                        entity='eddy_lab',
                        config={"epochs": num_epochs,
                                "Architecture": args.model,
                                "dataset": 'Pfam Seed'})

    # Testing loop
    
    dist.barrier() # Synchronize processes

    outputs = [None for _ in range(num_gpus)]

    batch_preds, n_batches = mu.test_step_batch(data_loader = data_loader,
                            clan_classifier = clan_classifier,
                            fam_classifier = fam_classifier,
                            device = rank,
                            data_utils = data_utils,
                            hmm_dict = test_dict,
                            fam = args.model=='fam')
        
    dist.gather(batch_preds, outputs)
    dist.barrier() # Synchronize processes

    # Log to wandb if master process

    if rank == 0:
        print(f'Epoch {epoch} Loss {epoch_loss} Validation: {validation_loss}')
        print('------------------------------------------------')
        
        if not args.no_log:
            wandb.log({'Epoch loss': epoch_loss, 
                       'Epoch Accuracy': epoch_accuracy,
                       'Epoch TPR': epoch_tpr,
                       'Epoch FPR': epoch_fpr,
                       'Validation Loss': validation_loss,
                       'Validation Accuracy':  val_accuracy,
                       'Validation TPR': val_tpr,
                       'Validation FPR': val_fpr,
                       'Validation 20 Loss': validation_loss_20,
                       'Validation 20 Accuracy':  val_accuracy_20,
                       'Validation 20 TPR': val_tpr_20,
                       'Validation 20 FPR': val_fpr_20,
                       'Learning rate': optimizer.param_groups[0]['lr']})

        if validation_loss < best_validation_loss:  # Check for better performing model
            best_validation_loss = validation_loss  # Update best validation loss
            # save model with epoch num in save path
            torch.save(classifier.state_dict(), save_path / f'epoch_{epoch}.pth')
            epochs_without_improvement = 0
        
        else:
            epochs_without_improvement += 1
        
        if epoch_loss < best_epoch_loss:  # Check for better performing model
            best_epoch_loss = epoch_loss  # Update best validation loss
            # save model with epoch num in save path
            torch.save(classifier.state_dict(), save_path / f'epoch_{epoch}.pth')
        
        if epochs_without_improvement >= 5:
            print('Early stopping triggered. No improvement in validation loss for 5 epochs.')
            break

    dist.barrier() # Synchronize processes
    
    cleanup() # Cleanup all spawned processes

    '''
    Need to save the batch_preds
    '''

    return