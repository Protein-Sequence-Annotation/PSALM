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

def loadAll(rank, num_gpus, args, save_path, train_dict_path=None, val_dict_path=None, test_dict_path=None):

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

    

    # Instantiate dataset helper object
    
    length_limit = 4096 # Covers 99.75% sequences
    esm_model_name =  f'esm2_{args.esm_size}_UR50D' # ESM 650 million parameter model

    data_utils = mu.DataUtils(args.root, esm_model_name, length_limit, rank, layer_num = args.layer_number) # DataUtils object

    # Model choice based on user input
    def load_modified_state_dict(model, state_dict_path, prefix="module._orig_mod."):
        """
        Load a state_dict into a model, removing a specified prefix from the keys if necessary.

        Args:
            model (torch.nn.Module): The model into which the state_dict will be loaded.
            state_dict_path (str): Path to the saved state_dict.
            prefix (str): The prefix to remove from the state_dict keys. Default is "module._orig_mod.".
            
        Returns:
            None: The function modifies the model in place.
        """
        # Load the state_dict from the file
        state_dict = torch.load(state_dict_path)

        # Create a new state_dict without the prefix
        new_state_dict = {key.replace(prefix, ""): value for key, value in state_dict.items()}
        # new_state_dict = {f"module.{key}": value for key, value in state_dict.items()}


        # Load the modified state_dict into the model
        model.load_state_dict(new_state_dict)

    if args.mode == 'clan': # Clan batch model
        classifier = cf.ClanLSTMbatch(data_utils.embedding_dim,data_utils.clan_count).to(rank)
    elif args.mode == 'fam': # fam batch model
        classifier = cf.FamLSTMbatch(data_utils.embedding_dim, data_utils.maps, rank).to(rank)
    elif args.mode == 'eval': # Evaluate full pipeline
        clan_classifier = cf.ClanLSTMbatch_onehot(data_utils.embedding_dim,data_utils.clan_count).to(rank)
        load_modified_state_dict(clan_classifier, args.clan_filename, prefix="module._orig_mod.")
        fam_classifier = cf.FamLSTMbatch(data_utils.embedding_dim, data_utils.maps, rank).to(rank)
        load_modified_state_dict(fam_classifier, args.fam_filename, prefix="module._orig_mod.")
    elif args.mode == 'only': # fam only model
        classifier = cf.FamLSTMbatch(data_utils.embedding_dim, data_utils.maps, rank).to(rank)
    else:
        print('Incorrect Model choice')
        sys.exit(2)
    
    if args.mode == 'eval':
        clan_classifier = torch.compile(clan_classifier, mode='max-autotune') # Pre compile for optimization
        fam_classifier = torch.compile(fam_classifier, mode='max-autotune') # Pre compile for optimization
        clan_classifier = DDP(clan_classifier, device_ids=[rank], output_device=rank, find_unused_parameters=False) # DDP
        fam_classifier = DDP(fam_classifier, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    else:
        classifier = torch.compile(classifier, mode='max-autotune') # Pre compile for optimization
        classifier = DDP(classifier, device_ids=[rank], output_device=rank, find_unused_parameters=False) # DDP

    if args.resume != 'none': # Load checkpoint if resuming training
        classifier_path = Path(f'{args.resume}')
        load_modified_state_dict(classifier, classifier_path, prefix="")

    # Load validation set if master process - will change once parallelized

    if not args.no_validation:
        # Match train and val suffix always!
        dataset = data_utils.get_dataset(mode='Val', suffix=args.suffix) # Get corresponding dataset from fasta
        val_loader = data_utils.get_dataloader(dataset, rank, num_gpus) # num_gpus = ? for validation

        # dataset = data_utils.get_dataset(mode='Val_20', suffix='') # Get corresponding dataset from fasta
        # val_loader_20 = data_utils.get_dataloader(dataset, rank, num_gpus) # num_gpus = ? for validation
    
    # Parameters for training loop
    mode = 'Test' if args.mode=="eval" else 'Train'
    dataset = data_utils.get_dataset(mode=mode, suffix=args.suffix) # Get corresponding dataset from fasta
    data_loader = data_utils.get_dataloader(dataset, rank, num_gpus) # Parallelize data loader
    
    if args.mode != 'eval':
        # Load scans for train and validation
        with open(train_dict_path, 'rb') as f:
            train_dict = pickle.load(f)
        
        with open(val_dict_path, 'rb') as f:
            val_dict = pickle.load(f)

        # with open(f'datasets/Val_20_targets.pkl', 'rb') as f:
        #     val_dict_20 = pickle.load(f)

        loss_fn = nn.CrossEntropyLoss() # fam loss is hard coded
        lr = args.learning_rate #/num_gpus # EFFECTIVE lr 1e-3 for train with L1,  1e-4 for train without L1, 1e-5 for fine tune
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        num_epochs = args.num_epochs
        # lower LR if less than 10% decrease - can change to schedule free optimizer
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=np.sqrt(0.1), patience=1, threshold=0, threshold_mode="abs")
        scheduler = mu.CustomReduceLROnPlateau(optimizer, 
                                               mode='max', 
                                               factor=np.sqrt(0.1), 
                                               patience=1, 
                                               threshold=0, 
                                               threshold_mode="abs", 
                                               cooldown=2,
                                               min_lr=0,
                                               eps=1e-8,
                                               verbose=False)
        
        # Initialize wandb if master process

        if not args.no_log and rank == 0:
            run = wandb.init(project=args.project, 
                            entity='eddy_lab',
                            config={"epochs": num_epochs,
                                    "lr": lr,
                                    "Architecture": args.mode,
                                    "dataset": 'Pfam Seed'})

        # Training loop
        
        best_validation_loss = 0.07486 #1e3 # Start with high loss
        best_val_accuracy = 0
        best_epoch_loss = 1e3
        epochs_without_improvement = 0 # for early stopping

        dist.barrier() # Synchronize processes

        for epoch in tqdm(range(num_epochs), total=num_epochs, desc='Epochs completed', disable= rank != 0):

            epoch_loss, epoch_accuracy, epoch_tpr, epoch_fpr = mu.train_step_batch(data_loader = data_loader,
                                    classifier = classifier,
                                    loss_fn = loss_fn,
                                    optimizer = optimizer,
                                    device = rank,
                                    data_utils = data_utils,
                                    hmm_dict = train_dict,
                                    fam = args.mode=='fam')
            
            dist.reduce(epoch_loss, dst=0, op=dist.ReduceOp.AVG) # Average epoch loss from all processes
            dist.reduce(epoch_accuracy, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(epoch_tpr, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(epoch_fpr, dst=0, op=dist.ReduceOp.AVG)
            dist.barrier() # Synchronize processes

            # Compute validation loss 

            if not args.no_validation:

                validation_loss,val_accuracy,val_tpr,val_fpr = mu.validate_batch(data_loader = val_loader,
                                                    classifier = classifier,
                                                    loss_fn = loss_fn,
                                                    device = rank,
                                                    data_utils = data_utils,
                                                    hmm_dict = val_dict,
                                                    fam = args.mode=='fam')

                dist.reduce(validation_loss, dst=0, op=dist.ReduceOp.AVG) # Average epoch loss from all processes
                dist.reduce(val_accuracy, dst=0, op=dist.ReduceOp.AVG) 
                dist.reduce(val_tpr, dst=0, op=dist.ReduceOp.AVG)
                dist.reduce(val_fpr, dst=0, op=dist.ReduceOp.AVG)
                dist.barrier() # Synchronize processes

                # validation_loss_20, val_accuracy_20, val_tpr_20, val_fpr_20 = mu.validate_batch_onlyfams_onehot(data_loader = val_loader_20,
                #                                     classifier = classifier,
                #                                     loss_fn = loss_fn,
                #                                     device = rank,
                #                                     data_utils = data_utils,
                #                                     hmm_dict = val_dict_20,
                #                                     fam = args.mode=='fam')

                # dist.reduce(validation_loss_20, dst=0, op=dist.ReduceOp.AVG) # Average epoch loss from all processes
                # dist.reduce(val_accuracy_20, dst=0, op=dist.ReduceOp.AVG) 
                # dist.reduce(val_tpr_20, dst=0, op=dist.ReduceOp.AVG)
                # dist.reduce(val_fpr_20, dst=0, op=dist.ReduceOp.AVG)
                # dist.barrier() # Synchronize processes
            else:
                validation_loss = 0.
                val_accuracy = 0.
                val_tpr = 0.
                val_fpr = 0.

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
                            # 'Validation 20 Loss': validation_loss_20,
                            # 'Validation 20 Accuracy':  val_accuracy_20,
                            # 'Validation 20 TPR': val_tpr_20,
                            # 'Validation 20 FPR': val_fpr_20,
                            'Learning rate': optimizer.param_groups[0]['lr']})

                if val_accuracy > best_val_accuracy:  # Check for better performing model
                    best_val_accuracy= val_accuracy  # Update best validation loss
                    # save model with epoch num in save path
                    # torch.save(classifier.module.state_dict(), save_path / f'{args.mode}_epoch_{epoch}.pth') #if no work, delete module part
                    # torch.save(classifier.module.state_dict(), save_path / f'epoch_{epoch}.pth') #if no work, delete module part
                    epochs_without_improvement = 0
                
                else:
                    epochs_without_improvement += 1
                
                if epoch_loss < best_epoch_loss:  # Check for better performing model
                    best_epoch_loss = epoch_loss  # Update best validation loss
                    # save model with epoch num in save path
                    # torch.save(classifier.state_dict(), save_path / f'{args.mode}_epoch_{epoch}.pth')
                    torch.save(classifier.state_dict(), save_path / f'{args.mode}_epoch_{epoch}.pth')
                
                if epochs_without_improvement >= 10:
                    print('Early stopping triggered. No improvement in validation loss for 5 epochs.')
                    break
            if not args.no_validation:
                scheduler.step(val_accuracy) # Scheduler step
            dist.barrier() # Synchronize processes
    
    # test
    else:
        with open(test_dict_path, 'rb') as f:
            test_dict = pickle.load(f)

        mu.test_step_batch(data_loader = data_loader,
                                clan_classifier = clan_classifier,
                                fam_classifier = fam_classifier,
                                device = rank,
                                data_utils = data_utils,
                                hmm_dict = test_dict,
                                fam = args.mode=='fam',
                                save_path = save_path)
        print(f"here now {rank}")
        sys.stdout.flush()
        dist.barrier()
        # if rank 0, merge the dictionaries f"{save_path}/gpu_{device}_preds.pkl"
        if rank == 0:
            all_preds = {}
            for i in range(num_gpus):
                with open(f"{save_path}/gpu_{i}_preds.pkl", 'rb') as f:
                    preds = pickle.load(f)
                    all_preds |= preds
                # delete the file
                os.remove(f"{save_path}/gpu_{i}_preds.pkl")
            with open(f"{save_path}/preds{args.suffix}.pkl", 'wb') as f:
                pickle.dump(all_preds, f)
        print(f"now here {rank}")
        sys.stdout.flush()
        # dist.barrier()
    
    cleanup() # Cleanup all spawned processes

    return