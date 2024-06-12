import wandb
from datetime import datetime
import argparse
import json
import yaml
import os
from torch.utils.data import DataLoader
from common.utils import DataCreator, save_pretrained, load_pretrained, cleanup, LpLoss, get_run_str_aug
from helpers.train_helper import pretraining_loop, training_loop, test_onestep_losses, test_unrolled_losses
from helpers.model_helper import get_dataloader, get_model, get_pretraining_model
from loss import *
from tqdm import tqdm

import random
import torch

def pretrain(
        args: argparse,
        epoch: int,
        model: torch.nn.Module,
        pretrain_optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        loader: DataLoader,
        data_creator: DataCreator,
        loss_fn,
        device: torch.cuda.device="cpu"
        ) -> None:
    """
    Pretraining Loop
    Args:
        args: command line inputs
        epoch: current epoch
        model: neural network PDE solver
        pretain_optimizer: optimizer used for training
        scheduler: scheduler
        loader: training dataloader
        data_creator: helper object to handle data
        loss_fn: criterion for training
        device: device (cpu/gpu)
    Returns:
        None
    """
    if args.verbose:
        print(f'Starting epoch {epoch}...')
    model.train()
    n_inner = data_creator.t_res
    for i in range(n_inner):
        epoch = epoch*n_inner + i
        losses = pretraining_loop(args, 
                                  epoch,
                                model, 
                                pretrain_optimizer, 
                                scheduler,
                                loader, 
                                data_creator, 
                                loss_fn,
                                device)
        if args.verbose:
            print(f'Training Loss (progress: {i / data_creator.t_res:.2f}): {losses[0]}')
        wandb.log({"pretrain/loss": losses[0]})

def train(args: argparse,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler,
          loader: DataLoader,
          data_creator: DataCreator,
          criterion: torch.nn.modules.loss,
          device: torch.cuda.device="cpu",
          name = "default") -> None:
    """
    Training loop.
    Args:
        args: command line inputs
        epoch: current epoch
        model: neural network PDE solver
        optimizer: optimizer used for training
        scheduler: scheduler
        loader: training dataloader
        data_creator: helper object to handle data
        loss_fn: criterion for training
        device: device (cpu/gpu)
    Returns:
        None
    """
    if args.verbose:
        print(f'Starting epoch {epoch}...')
    model.train()

    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    # Loop over every epoch as often as the number of timesteps in one trajectory.
    # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
    # Therefore in expectation the whole available training information is covered.
    for i in range(data_creator.t_res):
        losses = training_loop(model, 
                               optimizer, 
                               scheduler,
                               unrolling,
                               loader, 
                               data_creator, 
                               criterion, 
                               device)
        if(i%10 == 0 and args.verbose):
            print(f'Training Loss (progress: {i / data_creator.t_res:.2f}): {losses}')
        wandb.log({f"{name}/train_loss": losses})


def test(
        args: argparse,
        model: torch.nn.Module,
        loader: DataLoader,
        data_creator: DataCreator,
        criterion: torch.nn.modules.loss,
        device: torch.cuda.device="cpu",
        ) -> torch.Tensor:
    """
    Test routine

    Args:
        args: command line inputs
        model: neural network PDE solver
        loader: training dataloader
        data_creator: helper object to handle data
        criterion: criterion for test
        device: device (cpu/gpu)
    Returns:
        torch.Tensor: unrolled forward loss
    """
    model.eval()
    if(args.mode == 'next_step'): # Double check when using pushforward/temporal bundling
        losses = test_unrolled_losses(model=model,
                                  nr_gt_steps=args.nr_gt_steps,
                                  loader=loader,
                                  data_creator=data_creator,
                                  criterion=criterion,
                                  device=device,
                                  verbose=args.verbose)
    elif(args.mode == 'fixed_future'):
        losses = test_onestep_losses(model=model,
                                  loader=loader,
                                  data_creator=data_creator,
                                  criterion=criterion,
                                  device=device,
                                  verbose=args.verbose)

    else:
        raise ValueError("Mode not implemented")
        
    return losses

def main(args: argparse):
    # Set seeds
    args.seed = int(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Seed: {args.seed}")

    # Setup
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    run_str, augmentation = get_run_str_aug(args, timestring)
    device = args.device

    # Start run after we get proper name
    run = wandb.init(project="my-project-name",
                     name = run_str,
                     mode=args.wandb_mode,
                     config=vars(args))
    
    pretrained_save_path = f'checkpoints/pt/pretrained_{run_str}.pth'

    print("Pretraining Strategy: {}".format(args.pretraining))
    print("Mode: {}".format(args.mode))
    
    data_creator = DataCreator(  time_window=args.time_window,
                                 time_future=args.time_future,
                                 t_resolution=args.base_resolution[0],
                                 x_resolution=args.base_resolution[1],
                                 t_range=args.t_range,
                                 device = args.device,
                                 mode = args.mode,
                                 target = args.target)

    # Training
    criterion = LpLoss(2, 2)

    if args.pretraining != "None":
        # Pretraining Model
        train_loader = get_dataloader(args.train_path_pt, 
                                  args, 
                                  mode='train', 
                                  pretraining=True, 
                                  augmentation=augmentation, 
                                  num_samples=args.num_samples_pt)
        loss_fn, pretraining_model, pretraining_optimizer, pretraining_scheduler = get_pretraining_model(args,
                                                                                   device, train_loader, num_aug=len(augmentation))
        
        print(pretraining_model)
        
        # Pretrain Model
        for epoch in tqdm(range(args.pretraining_epochs)):
            if args.pretraining == "transfer":
                train(args,
                        epoch, 
                        pretraining_model, 
                        pretraining_optimizer, 
                        pretraining_scheduler, 
                        train_loader,
                        data_creator, 
                        loss_fn, 
                        device=device,
                        name="pretrain")
            else:
                pretrain(args=args, 
                        epoch=epoch,
                        model=pretraining_model, 
                        pretrain_optimizer=pretraining_optimizer,
                        scheduler=pretraining_scheduler,
                        loader=train_loader,
                        data_creator=data_creator,
                        device=device, 
                        loss_fn=loss_fn)
    
        # Save and eval performance
        save_pretrained(pretraining_model, pretrained_save_path)
        # Delete and reload training data/models
        cleanup(args, train_loader, None, pretraining_model, pretraining_optimizer, pretraining_scheduler)
    
    subset_list = args.subset_list
    distribution_list = args.distribution_list
    samples_list = args.samples_list

    # Run fine-tuning experiments
    for distribution in distribution_list:
        for subset in subset_list:
            for ns in samples_list:
                # Reset seeds
                torch.manual_seed(args.seed)
                random.seed(args.seed)
                np.random.seed(args.seed)

                print(f"Finetuning on Number of samples: {ns}, Seed: {args.seed}, equation: {subset}, distribution: {distribution}")
                prefix = f"{ns}_{subset}_{distribution}"
                os.makedirs(f"checkpoints/{prefix}", exist_ok=True) 
                save_path_unique = f'checkpoints/{prefix}/{run_str}.pth'

                train_loader = get_dataloader(args.train_path if distribution == "in" else args.train_path_out, 
                                            args, 
                                            mode='train', 
                                            pretraining=False, 
                                            augmentation=augmentation, 
                                            num_samples=ns,
                                            subset=subset)
                valid_loader = get_dataloader(args.valid_path if distribution == "in" else args.valid_path_out, 
                                                args, 
                                                mode='valid',
                                                pretraining=False,
                                                subset=subset)
                model, optimizer, scheduler = get_model(args, device, train_loader)

                if(isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)):
                    scheduler.total_steps = len(train_loader)*args.num_epochs*data_creator.t_res
                    print("TOTAL STEPS: {}".format(scheduler.total_steps))

                print(model)
                if args.pretraining != "None":
                    model = load_pretrained(args, model, pretrained_save_path)

                min_val_loss = 10e10

                for epoch in tqdm(range(args.num_epochs)):
                    train(args, 
                        epoch, 
                        model, 
                        optimizer, 
                        scheduler, 
                        train_loader,
                        data_creator, 
                        criterion, 
                        device=device,
                        name=prefix)

                    val_loss = test(args, 
                                    model, 
                                    valid_loader, 
                                    data_creator, 
                                    criterion, 
                                    device=device)
                    
                    if args.verbose:
                        print(f"Validation Loss: {val_loss}\n")

                    wandb.log({
                        f"{prefix}/val_loss": val_loss,
                    })

                    if(val_loss < min_val_loss):
                        # Save model
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'model_optimizer_state_dict': optimizer.state_dict(),
                            'model_scheduler_state_dict': scheduler.state_dict(),
                            'loss': val_loss,
                        }

                        torch.save(checkpoint, save_path_unique)
                        if args.verbose:
                            print(f"Saved model at {save_path_unique}\n")
                            
                        min_val_loss = val_loss

                wandb.log({
                    f"{prefix}/min_val_loss": min_val_loss,
                })
                cleanup(args, train_loader, valid_loader, model, optimizer, scheduler)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PDE Solver')

    ################################################################
    #  GENERAL
    ################################################################
    # General
    parser.add_argument('--config', type=str, help='Load settings from file in json format. Command line options override values in file.')
    parser.add_argument('--device', type=str, help='Device to run on')
    parser.add_argument('--model', type=str, help='model architecture')
    parser.add_argument('--batch_size', type=int, help='Batch Size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--wandb_mode', type=str, help='Wandb mode')
    parser.add_argument('--verbose', type=eval, help='Verbose')
    parser.add_argument('--seed', type=str, help='Seed')
    parser.add_argument('--mode', type=str, help='Mode')
    parser.add_argument('--mode_pt', type=str, help='Mode for pretraining')
    parser.add_argument('--run_all', type=eval, help='Run all seeds')

    # Pretraining
    parser.add_argument('--pretraining', type=str, 
        help='Select None for no pretraining, picl, piano, or combinatorial, sort_time, sort_space, jigsaw, axial_jigsaw, oddoneout')
    parser.add_argument('--pretraining_epochs', type=int, help='num_epochs for pretraining')
    parser.add_argument('--pretrained_path', type=str, help='Path to pretrained model')

    # Handl contrastive_loss and similarity_fn based on pretraining strategy.
    parser.add_argument('--contrastive_loss', type=str, help='If using picl')
    parser.add_argument('--similarity_fn', type=str, help='f')
    parser.add_argument("--mask_ratio", type=float, help="Mask ratio for masked pretraining")

    ################################################################
    #  Experiments
    ################################################################

    parser.add_argument('--samples_list', type=lambda s: [int(item) for item in s.split('/')],
            default=[100, 250, 500, 1000], help="PDE base resolution on which network is applied")
    parser.add_argument('--distribution_list', type=lambda s: [str(item) for item in s.split('/')],
        default=["in", "out"], help="PDE base resolution on which network is applied")
    parser.add_argument('--subset_list', type=lambda s: [str(item) for item in s.split('/')],
        default=["adv", "heat", "burger"], help="PDE base resolution on which network is applied")

    ################################################################
    #  DATA
    ################################################################

    parser.add_argument('--train_path', type=str, help='Path to training data')
    parser.add_argument('--valid_path', type=str, help='Path to validation data')
    parser.add_argument('--load_all', type=eval, help='Load all data into memory')
    parser.add_argument('--parameter_ablation', type=eval, help="Add params to solver")
    parser.add_argument('--num_samples', type=int, help="Number of samples to load")
    parser.add_argument('--subset', type=str, help="Subset of data to use")
    parser.add_argument('--unrolling', type=int, help="Unrolling steps")
    parser.add_argument('--time_window', type=int, help="Time window")
    parser.add_argument('--time_future', type=int, help="Time future")
    
    ################################################################
    #  models
    ################################################################
    # FNO parameters
    parser.add_argument('--fno_modes', type=int, help='Number of modes for FNO')
    parser.add_argument('--fno_width', type=int, help='Width of FNO')
    parser.add_argument('--fno_num_layers', type=int, help='Number of layers for FNO')

    # Head
    parser.add_argument('--mlp', type=str, help='MLP spec for projector')
    parser.add_argument('--lr', type=float, help='Learning rate for model')
    parser.add_argument('--lr_min', type=float, help='Minimum learning rate for model')

    args = parser.parse_args()

    # Load args from config
    if args.config:
        filename, file_extension = os.path.splitext(args.config)
        # Load yaml
        if file_extension=='.yaml':
            t_args = argparse.Namespace()
            t_args.__dict__.update(yaml.load(open(args.config), Loader=yaml.FullLoader))
            args = parser.parse_args(namespace=t_args)
        elif file_extension=='.json':
            with open(args.config, 'rt') as f:
                t_args = argparse.Namespace()
                t_args.__dict__.update(json.load(f))
                args = parser.parse_args(namespace=t_args)
        else:
            raise ValueError("Config file must be a .yaml or .json file")
        

    main(args)
    
