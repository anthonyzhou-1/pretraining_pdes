import torch
from torch import nn
from typing import Tuple
from collections import OrderedDict
import argparse
import yaml
import json
import os
from common.augmentation import *

class DataCreator(nn.Module):
    """
    Helper class to construct input data and labels.
    """
    def __init__(
            self,
            time_window: int = 5,
            time_future: int = 5,
            t_resolution: int = 250,
            x_resolution: int =100,
            t_range: list = [-5, 5],
            device: str = 'cpu',
            mode: str='next_step',
            target: int=-1,
            ) -> None:
        """
        Initialize DataCreator class
        Args:
            pde (PDE): PDE at hand [CE, WE, ...]
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
        Returns:
            None
        """
        super().__init__()
        self.tw = time_window
        self.tf = time_future
        self.t_res = t_resolution
        self.x_res = x_resolution
        self.t_range = t_range
        self.mode = mode

        if(self.mode == 'fixed_future'):
            self.t_res = 1
            self.target = target
            print(f"Target: {self.target}")

        assert isinstance(self.tw, int)

    def create_data(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        if(self.mode == 'fixed_future'):
            return datapoints[:,:self.tw], datapoints[:,self.target-1].unsqueeze(1)

        shape = list(datapoints.shape)
        shape[0] = 0
        shape[1] = self.tw

        data = torch.empty(shape, device=datapoints.device)

        shape[1] = self.tf
        labels = torch.empty(shape, device=datapoints.device)
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            l = dp[step:self.tf + step]
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels
    
    def create_rand_data(self, datapoints: torch.Tensor, steps: list):
        shape = list(datapoints.shape)
        shape[0] = 0
        shape[1] = self.tw

        data = torch.empty(shape, device=datapoints.device)

        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            data = torch.cat((data, d[None, :]), 0)

        return data

def process_dict(state_dict: OrderedDict, prefix: str) -> OrderedDict:
    '''
    Processes state dict to remove prefixes
    '''

    return {k.partition(f'{prefix}.')[2]:state_dict[k] for k in state_dict.keys()}

def dict2tensor(d: dict) -> torch.Tensor:
    """
    Converts a dictionary to a tensor
    Args:
        d (dict): dictionary
    Returns:
        t (torch.Tensor): tensor
    """
    tensors = []
    for k, v in d.items():
        tensors.append(v.unsqueeze(0))
    return torch.transpose(torch.cat(tensors, dim=0), 0, 1)

def load_args(args, parser):
    # Load args from config
    if args.config:
        filename, file_extension = os.path.splitext(args.config)
        # Load yaml
        if file_extension=='.yaml':
            t_args = argparse.Namespace()
            t_args.__dict__.update(yaml.load(open(args.config), Loader=yaml.FullLoader))
            args = parser.parse_args(args=[], namespace=t_args)
        elif file_extension=='.json':
            with open(args.config, 'rt') as f:
                t_args = argparse.Namespace()
                t_args.__dict__.update(json.load(f))
                args = parser.parse_args(args=[], namespace=t_args)
        else:
            raise ValueError("Config file must be a .yaml or .json file")
    return args

def save_pretrained(model, save_path):
    if hasattr(model, 'encoder'):
        pretrained_checkpoint = {
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': model.encoder.state_dict(),
        }
    else:
        pretrained_checkpoint = {
            'model_state_dict': model.state_dict(),
        }

    torch.save(pretrained_checkpoint, save_path)
    print(f"Saved model at {save_path}\n")

def load_pretrained(args, model, save_path):
    if (args.pretraining == 'piano' or \
            args.pretraining == 'jigsaw' or \
            args.pretraining == 'sort_space' or \
            args.pretraining == 'sort_time' or \
            args.pretraining=='oddoneout' or \
            args.pretraining=='regression' or \
            args.pretraining=='derivative' or \
            args.pretraining=='combined' or \
            args.pretraining=='masked'):

            model.load_state_dict(torch.load(save_path)['encoder_state_dict'])
            print(f"Loaded encoder state dict from {save_path}")
    else:
        model.load_state_dict(torch.load(save_path)['model_state_dict'])
        print(f"Loaded model state dict from {save_path}")
    
    return model

def load_pretrained_from_dict(args, model, seed=1):
    model_path = args.__dict__[args.model][args.pretraining][seed]
    path = f"./checkpoints/pretrained_{model_path}.pth"

    model = load_pretrained(args, model, path)
    return model 

def cleanup(args, train_loader, valid_loader, model, optimizer, scheduler):
    print("torch.cuda.max_memory_allocated: %fGB"%(torch.cuda.max_memory_allocated(device=args.device)/1024/1024/1024))
    train_loader.dataset.cleanup()
    if valid_loader is not None:
        valid_loader.dataset.cleanup()
        del valid_loader
    del model
    del optimizer
    del scheduler
    del train_loader

    with torch.cuda.device(args.device):
        torch.cuda.empty_cache()

    print("torch.cuda.max_memory_allocated: %fGB"%(torch.cuda.max_memory_allocated(device=args.device)/1024/1024/1024))
    print("\n")

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms
        
    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
def get_run_str_aug(args, timestring):
    if(args.mode == 'fixed_future'):
        if('OOD' in args.train_path):
            if(args.augmentation_ratio == 0):
                augmentation = []
                run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_OOD_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}__{timestring}'
            else:
                if(args.augmentation == 'noise'):
                    augmentation = [NoiseInjection(args.noise_level)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_OOD_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_pt{args.noise_level}_{timestring}'
                elif(args.augmentation == 'shift'):
                    augmentation = [LinearShift(args.shift)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_OOD_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_shift{args.shift}_{timestring}'
                elif(args.augmentation == 'scale'):
                    augmentation = [Scale(args.shift)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_OOD_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_scale{args.max_scale}_{timestring}'
                elif(args.augmentation == 'both'):
                    augmentation = [LinearShift(args.shift), NoiseInjection(args.noise_level)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_OOD_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_both_pt{args.noise_level}_shift{args.shift}_{timestring}'

                elif(args.augmentation == 'both'):
                    augmentation = [LinearShift(args.shift), NoiseInjection(args.noise_level),
                                    Scale(args.max_scale)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_OOD_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_all_pt{args.noise_level}_shift{args.shift}_scale{args.max_scale}_{timestring}'
        else:
            if(args.augmentation_ratio == 0):
                augmentation = []
                run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}__{timestring}'
            else:
                if(args.augmentation == 'noise'):
                    augmentation = [NoiseInjection(args.noise_level)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_pt{args.noise_level}_{timestring}'
                elif(args.augmentation == 'shift'):
                    augmentation = [LinearShift(args.shift)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_shift{args.shift}_{timestring}'
                elif(args.augmentation == 'scale'):
                    augmentation = [Scale(args.shift)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_scale{args.max_scale}_{timestring}'
                elif(args.augmentation == 'both'):
                    augmentation = [LinearShift(args.shift), NoiseInjection(args.noise_level)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_both_pt{args.noise_level}_shift{args.shift}_{timestring}'

                elif(args.augmentation == 'both'):
                    augmentation = [LinearShift(args.shift), NoiseInjection(args.noise_level),
                                    Scale(args.max_scale)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_{args.mode}_{args.target}_{args.num_samples_pt}_{args.num_samples}_{args.model}_{args.pretraining}_all_pt{args.noise_level}_shift{args.shift}_scale{args.max_scale}_{timestring}'
    else:
        if(args.unrolling > 0):
            if(args.augmentation_ratio == 0):
                augmentation = []
                run_str = f'{args.seed}_{args.description}_{args.pde}_pushforward_NEW_DOWNSAMPLED_{args.mode}_{args.num_samples}_{args.model}_{args.pretraining}__{timestring}'
            else:
                if(args.augmentation == 'noise'):
                    augmentation = [NoiseInjection(args.noise_level)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_pushforward_NEW_DOWNSAMPLED_{args.mode}_{args.num_samples}_{args.model}_{args.pretraining}_pt{args.noise_level}_{timestring}'
                elif(args.augmentation == 'shift'):
                    augmentation = [LinearShift(args.shift)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_pushforward_NEW_DOWNSAMPLED_{args.mode}_{args.num_samples}_{args.model}_{args.pretraining}_shift{args.shift}_{timestring}'

                elif(args.augmentation == 'galileo'):
                    augmentation = [Galileo(args.max_velocity)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_pushforward_NEW_DOWNSAMPLED_{args.mode}_{args.num_samples}_{args.model}_{args.pretraining}_galileo{args.max_velocity}_{timestring}'
                elif(args.augmentation == 'scale'):
                    augmentation = [Scale(args.max_scale)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_pushforward_NEW_DOWNSAMPLED_{args.mode}_{args.num_samples}_{args.model}_{args.pretraining}_scale{args.max_scale}_{timestring}'

                elif(args.augmentation == 'both'):
                    augmentation = [LinearShift(args.shift), NoiseInjection(args.noise_level)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_pushforward_NEW_DOWNSAMPLED_{args.mode}_{args.num_samples}_{args.model}_{args.pretraining}_both_pt{args.noise_level}_shift{args.shift}_{timestring}'

                elif(args.augmentation == 'all'):
                    augmentation = [LinearShift(args.shift), NoiseInjection(args.noise_level),
                                    Scale(args.max_scale)]
                    run_str = f'{args.seed}_{args.description}_{args.pde}_pushforward_NEW_DOWNSAMPLED_{args.mode}_{args.num_samples}_{args.model}_{args.pretraining}_all_pt{args.noise_level}_shift{args.shift}_scale{args.max_scale}_{timestring}'
        else:
            if(args.augmentation_ratio == 0):
                augmentation = []
                run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_{args.mode}_{args.num_samples}_{args.model}_{args.pretraining}__{timestring}'
            else:
                augmentation = [NoiseInjection(args.noise_level)]
                run_str = f'{args.seed}_{args.description}_{args.pde}_NEW_DOWNSAMPLED_{args.mode}_{args.num_samples}_{args.model}_{args.pretraining}_pt{args.noise_level}_{timestring}'
    if(len(args.subset) < 8):
        print("Updating run string.")
        if("OOD" in run_str):
            split_run_str = run_str.split("_")
            split_run_str.insert(14, args.subset)
            run_str = "_".join(split_run_str)
        else:
            split_run_str = run_str.split("_")
            split_run_str.insert(13, args.subset)
            run_str = "_".join(split_run_str)
    return run_str, augmentation


def load_pretrained_path(args, pretraining_model, run_str):
    # Can reuse pretrained model for different fine-tune data sizes
    split_run_str = run_str.split("_")[:-1]
    print(len(split_run_str))

    # Pretrained model comes from whole pretraining set, remove subset name from load string
    if(len(split_run_str) == 16): # Navier-Stokes case
        split_run_str = split_run_str[:13] + split_run_str[14:]
    if(len(split_run_str) == 17): # OOD case
        split_run_str = split_run_str[:7] + split_run_str[8:]
        split_run_str = split_run_str[:13] + split_run_str[14:]

    split_run_str = split_run_str[:11] + split_run_str[12:]
    pretrained_path = "./pretrained_models/" + '_'.join(split_run_str)

    run_pretraining = True
    if(args.load_pretrained):
        print("\nLOADING THE PRETRAINED MODEL")
        print(pretrained_path)
        try:
            pretraining_model.load_state_dict(torch.load(pretrained_path))
            run_pretraining = False
        except FileNotFoundError:
            print("NO PRETRAINED MODEL FOUND.")
            
    return pretraining_model, run_pretraining