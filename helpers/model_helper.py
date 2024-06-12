import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fno1d import FNO1d_bundled
from models.fno2d import FNO2d_bundled, PIANO_FNO, FNO_pretrain
from models.deeponet import DeepONet1D, DeepONet2D
from models.oformer import OFormer1D, Encoder1D, PointWiseDecoder1D
from models.oformer import OFormer2D, SpatialTemporalEncoder2D, PointWiseDecoder2D
from models.unet import Unet

from common.datasets import PDEDataset, PDEDataset2D
from common.utils import LpLoss
from loss import *
from loss.spatiotemporal_losses import *
from loss.statistical_losses import Derivative_Regression, Coefficient_Regression, Masked_Regression

def get_dataloader(path, args, mode='train', augmentation=[], pretraining=False, testing=False, num_samples=-1, subset="heat,adv,burger"):
    if args.pde_dim == 1:
        dataset = PDEDataset(path, 
                            pde=args.pde, 
                            mode=mode, 
                            resolution=args.base_resolution,
                            augmentation=augmentation,
                            augmentation_ratio=args.augmentation_ratio,
                            load_all=args.load_all,
                            device=args.device,
                            subset=subset,
                            num_samples=num_samples,
                            )
    elif args.pde_dim == 2:
        dataset = PDEDataset2D(path, 
                            pde=args.pde, 
                            mode=mode,
                            resolution=args.base_resolution,
                            augmentation=augmentation,
                            augmentation_ratio=args.augmentation_ratio,
                            load_all=args.load_all,
                            device=args.device,
                            subset=subset,
                            num_samples=num_samples,)
    else:
        raise ValueError("PDE dimension should be 1 or 2")
        
    # Different batch size if we're pretraining
    batch_size = args.pretraining_batch_size if pretraining else args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0,
                            shuffle=True if (mode == 'train' or testing) else False)
    
    return dataloader

def get_backbone(args, device, pretraining=False):
    eq_variables = {}
    
    if args.model == "FNO1D":
        model = FNO1d_bundled(
            time_window=args.time_window,
            time_future=args.time_future,
            modes=args.fno_modes,
            width=args.fno_width,
            num_layers=args.fno_num_layers,
            eq_variables=eq_variables
            ).to(device)
        print("Initialized FNO1D model")
        
    elif args.model == "FNO2D":
        if(args.pretraining == "piano"):
            if pretraining:
                model = FNO_pretrain(modes=args.fno_modes, tw_in=args.time_window, width=args.fno_width,
                                    h_size=args.pretraining_h_size).to(device)
            else:
                model = PIANO_FNO(modes1=args.fno_modes,
                                modes2=args.fno_modes,
                                width=args.fno_width,
                                d_embds=32).to(device)
        else:
            model = FNO2d_bundled(time_window=args.time_window,
                                  time_future=args.time_future,
                                  modes1=args.fno_modes,
                                  modes2=args.fno_modes,
                                  width=args.fno_width,
                                  num_layers=args.fno_num_layers,
                                  eq_variables=eq_variables).to(device)

        print("Initialized FNO2D model")

    elif args.model == "DeepONet1D":
        model = DeepONet1D(
            time_window=args.time_window,
            time_future=args.time_future,
            layer_sizes_branch=args.layer_sizes_branch,
            layer_sizes_trunk=args.layer_sizes_trunk,
            activation=args.activation,
            kernel_initializer=args.kernel_initializer,
            regularization=args.regularization,
            eq_variables=eq_variables,
            seed=args.seed,
            ).to(device)

    elif args.model == "DeepONet2D":
        model = DeepONet2D(
            time_window=args.time_window,
            time_future=args.time_future,
            layer_sizes_branch=args.layer_sizes_branch,
            layer_sizes_trunk=args.layer_sizes_trunk,
            activation=args.activation,
            kernel_initializer=args.kernel_initializer,
            regularization=args.regularization,
            eq_variables=eq_variables,
            ).to(device)

    elif args.model == "OFormer1D":
        oformer_encoder = Encoder1D(
            input_channels=args.time_window+len(eq_variables)+2,
            in_emb_dim=args.in_emb_dim,
            out_seq_emb_dim=args.out_seq_emb_dim,
            depth=args.depth,
            dropout=args.dropout,
            res=args.enc_res
            ).to(device)
        oformer_decoder = PointWiseDecoder1D(
            out_channels=1,
            latent_channels=args.latent_channels,
            decoding_depth=args.decoding_depth,
            scale=args.scale,
            res=args.dec_res
            ).to(device)
        model = OFormer1D(
            oformer_encoder, 
            oformer_decoder, 
            time_future=args.time_future, 
            eq_variables=eq_variables
            ).to(device)

    elif args.model == "OFormer2D":
        oformer_encoder = SpatialTemporalEncoder2D(
            input_channels=args.time_window + len(eq_variables) + 5, #TODO: Fix
            in_emb_dim=args.in_emb_dim,
            out_seq_emb_dim=args.out_seq_emb_dim,
            heads=args.heads,
            depth=args.depth
            ).to(device)
        oformer_decoder = PointWiseDecoder2D(
            out_channels=1,
            latent_channels=args.latent_channels,
            propagator_depth=args.decoding_depth,
            scale=args.scale,
            out_steps=args.time_future
            ).to(device)
        model = OFormer2D(
            oformer_encoder, 
            oformer_decoder,
            time_future=args.time_future, 
            eq_variables=eq_variables
            ).to(device)

    elif args.model == "Unet":
        model = Unet(dim = 16, 
                     init_dim=16, 
                     out_dim=args.time_future, 
                     channels=args.time_window + len(eq_variables) + 2, 
                     with_time_emb=False, 
                     resnet_block_groups=8, 
                     dim_mults=(1, 2, 4),
                     eq_variables=eq_variables).to(device)
    else:
        raise ValueError("model not found")
    
    return model

def get_model(args, device, loader):
    model = get_backbone(args, device, pretraining=False)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of model parameters: {params}')

    if("FNO" not in args.model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # weight decay does not work w/ FNO
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.model == "OFormer1D":
        steps_per_epoch = len(loader)*args.base_resolution[0]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr_max,
                                                                epochs=args.num_epochs, steps_per_epoch=steps_per_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 
                                                               eta_min=args.lr_min)
        

    return model, optimizer, scheduler

def get_pretraining_model(args, device, loader, num_aug=0):

    # Setup pretraining models
    model = get_backbone(args, device, pretraining=True)
    loss_fn = None

    ### Contrastive Pretraining Losses
    if((args.pretraining == 'picl') and ('1D' in args.model)):
        loss_fn = GCL(device=device, tau=args.pretraining_tau)
    elif((args.pretraining == 'picl') and ('2D' in args.model)):
        loss_fn = GCL2D(device=device, tau=args.pretraining_tau)
    elif(args.pretraining == 'piano'):
        loss_fn = NT_Xent(args.pretraining_batch_size, args.pretraining_temperature)
    elif(args.pretraining == 'combinatorial'):
        loss_fn = nn.MSELoss(reduction='mean')
    
    ## Transfer learning
    elif(args.pretraining == "transfer"):
        loss_fn = LpLoss(2, 2)

    ### Video Pretraining Losses
    elif(args.pretraining == 'jigsaw'):
        model = Jigsaw(config=args, model=model).to(device)
    elif(args.pretraining == 'sort_time'):
        model = Sort(config=args, model=model).to(device)
    elif(args.pretraining == 'sort_space'):
        model = Sort(config=args, model=model).to(device)
    elif(args.pretraining == 'oddoneout'):
        model = Binary_Sort(config=args, model=model).to(device)

    # Regression/Reconstruction Losses
    elif(args.pretraining == 'derivative'):
        model = Derivative_Regression(args, model, loader.dataset.dx.item(), loader.dataset.dx.item(), loader.dataset.dt.item()).to(device)
    elif(args.pretraining == 'regression'):
        model = Coefficient_Regression(args, model).to(device)
    elif(args.pretraining == "masked"):
        model = Masked_Regression(args, model).to(device)
    else:
        raise ValueError("Pretraining model not found")
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of pretraining model parameters: {params}')

    # Get Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretraining_lr)

    # Get scheduler
    if args.model == "OFormer1D":

        steps_per_epoch = len(loader)*args.base_resolution[0]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                epochs=args.pretraining_epochs, steps_per_epoch=steps_per_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pretraining_epochs,
                                                                       eta_min=args.lr_min)

    return loss_fn, model, optimizer, scheduler
