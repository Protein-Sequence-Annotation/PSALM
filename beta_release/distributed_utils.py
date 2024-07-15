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
from numpy import sqrt

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

def loadAll(rank, num_gpus, args, save_path, train_dict_path, val_dict_path):

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
    with open(train_dict_path, 'rb') as f:
        train_dict = pickle.load(f)
    
    with open(val_dict_path, 'rb') as f:
        val_dict = pickle.load(f)

    # Instantiate dataset helper object
    
    length_limit = 4096 # Covers 99.75% sequences
    esm_model_name =  'esm2_t33_650M_UR50D' # ESM 650 million parameter model

    data_utils = mu.DataUtils(args.root, esm_model_name, length_limit, rank)
    
    # Model choice based on user input

    if args.model == 'clan': # Clan batch model
        classifier = cf.ClanLSTMbatch(data_utils.embedding_dim,data_utils.clan_count).to(rank)
    elif args.model == 'fam': # fam batch model
        classifier = cf.FamLSTMbatch(data_utils.embedding_dim, data_utils.maps, rank).to(rank)
    else:
        print('Incorrect Model choice')
        sys.exit(2)

    classifier = torch.compile(classifier, mode='max-autotune') # Pre compile for optimization
    classifier = DDP(classifier, device_ids=[rank], output_device=rank, find_unused_parameters=False) # DDP

    if args.resume != 'none': # Load checkpoint if resuming training
        classifier_path = Path(f'{args.resume}')
        classifier.load_state_dict(torch.load(classifier_path))

    # Load validation set if master process - will change once parallelized

    if not args.no_validation:
        dataset = data_utils.get_dataset(mode='Validation', suffix='FINAL') # Get corresponding dataset from fasta
        val_loader = data_utils.get_dataloader(dataset, rank, num_gpus) # num_gpus = ? for validation
    
    # Parameters for training loop

    loss_fn = nn.CrossEntropyLoss() # fam loss is hard coded
    lr = args.learning_rate #/num_gpus # EFFECTIVE lr 1e-3 for train with L1,  1e-4 for train without L1, 1e-5 for fine tune
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr) #added weight decay
    # lower LR if less than 10% decrease - can change to schedule free optimizer
    after_warmup_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.1, threshold_mode="rel")
    # Initialize CustomLR with the optimizer, warmup steps, and the after-warmup scheduler
    scheduler = mu.CustomLR(optimizer, args.warmup_steps, after_warmup_scheduler)

    num_epochs = args.num_epochs

    # Initialize wandb if master process

    if not args.no_log and rank == 0:
        run = wandb.init(project=args.project, 
                        entity='eddy_lab',
                        config={"epochs": num_epochs,
                                "lr": lr,
                                "Architecture": "Fam",
                                "dataset": 'Pfam Seed'})

    # Training loop
    
    dataset = data_utils.get_dataset(mode='Train', suffix='FINAL') # Get corresponding dataset from fasta
    data_loader = data_utils.get_dataloader(dataset, rank, num_gpus) # Parallelize data loader

    best_validation_loss = 1e3 # Start with high loss
    dist.barrier() # Synchronize processes
    
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc='Epochs completed', disable= rank != 0):

        epoch_loss = mu.train_step_batch(data_loader = data_loader,
                                classifier = classifier,
                                loss_fn = loss_fn,
                                optimizer = optimizer,
                                device = rank,
                                data_utils = data_utils,
                                hmm_dict = train_dict,
                                fam = args.model=='fam')
        
        dist.reduce(epoch_loss, dst=0, op=dist.ReduceOp.AVG) # Average epoch loss from all processes
        dist.barrier() # Synchronize processes

        # Compute validation loss

        if not args.no_validation:

            validation_loss = mu.validate_batch(data_loader = val_loader,
                                                classifier = classifier,
                                                loss_fn = loss_fn,
                                                device = rank,
                                                data_utils = data_utils,
                                                hmm_dict = val_dict,
                                                fam = args.model=='fam')

            dist.reduce(validation_loss, dst=0, op=dist.ReduceOp.AVG) # Average epoch loss from all processes
            dist.barrier() # Synchronize processes
        else:
            validation_loss = 0.

        # Log to wandb if master process

        if rank == 0:
            print(f'Epoch {epoch} Loss {epoch_loss} Validation: {validation_loss}')
            print('------------------------------------------------')
            
            if not args.no_log:
                wandb.log({'Epoch loss': epoch_loss, 'Validation Loss': validation_loss, 'Learning rate': optimizer.param_groups[0]['lr']})

            if validation_loss < best_validation_loss: # Check for better performing model
                best_validation_loss = validation_loss # Update best validation loss
                torch.save(classifier.state_dict(), save_path / f'epoch_{epoch}.pth') # Save checkpoint

        scheduler.step(validation_loss) # Scheduler step
        dist.barrier() # Synchronize processes

    
    
    cleanup() # Cleanup all spawned processes

    return