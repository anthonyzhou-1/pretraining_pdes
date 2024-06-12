import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from common.augmentation import *
import random


class PDEDataset(Dataset):
    """Load samples of a PDE Dataset, get items according to PDE"""

    def __init__(
            self,
            path: str,
            pde: str,
            mode: str,
            resolution: list = None,
            augmentation = [],
            augmentation_ratio: float = 0.0,
            shift: str = 'fourier',
            load_all: bool = False,
            device: str = 'cuda:0',
            ) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.resolution = resolution or (250, 100)
        self.data = f[self.mode]

        self.variables = self.get_variables(pde)
        self.attrs = ['u'] + self.variables
        for attr in self.attrs + ['x', 't']:
            setattr(self, attr, 
                    torch.as_tensor(np.array(self.data[attr]), 
                                    dtype=torch.float32, 
                                    device=device if load_all else 'cpu')
                    )

        f.close()
        self.indexes = torch.arange(len(self.t))
        
        self.augmentation = augmentation
        self.shift = shift
        self.augmentation_ratio = augmentation_ratio

        self.device = device

        self.dt = self.t[1] - self.t[0]
        self.dx = self.x[1] - self.x[0]

    def get_variables(self, pde):
        if pde == "kdv_burgers":
            return ['alpha', 'beta', 'gamma']
        elif pde == "ks":
            return ["A", 'omerga']
        elif pde == "advection":
            return ["a"]
        else:
            raise ValueError("PDE not found")

    def __len__(self):
        return len(self.indexes)*(len(self.augmentation)+1)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        idx = self.indexes[i]
        # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
        t_idx = i % (len(self.augmentation) + 1)
        idx = i // (len(self.augmentation) + 1)

        variables = {attr: getattr(self, attr)[idx] for attr in self.attrs}
        u = variables.pop('u')
        x = self.x
        t = self.t

        if self.mode == "train" and self.augmentation is not None:
            if self.augmentation_ratio > random.random(): # augment data w/ probability augmentation_ratio
                # Augment data
                X = to_coords(x, t)
                sol = (u, X)
                sol = self.augmentation[t_idx](sol, self.shift)
                u = sol[0]

        return u, x, variables
    
    def choose_subset(self):
        raise NotImplemented
        

class PDEDataset2D(Dataset):
    """Load samples of a 2D PDE Dataset, get items according to PDE"""

    def __init__(
            self,
            path: str,
            pde: str,
            mode: str,
            resolution: list = None,
            augmentation = None,
            augmentation_ratio: float = 0.0,
            shift: str = 'fourier',
            load_all: bool = True,
            device: str = 'cuda:0',
            spatial_downsampling=1,
            temporal_horizon=32,
            subset = None,
            num_samples = -1
            ) -> None:

        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            mode: [train, valid, test]
            resolution: base resolution of the dataset [nt, nx, ny]
            augmentation: Data augmentation object
            augmentation_ratio: Probability to augment data
            load_all: load all the data into memory
            device: if load_all, load data onto device
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.resolution = resolution or (100, 64, 64)
        self.data = f[self.mode]
        self.spatial_downsampling = spatial_downsampling
        self.temporal_horizon = temporal_horizon
        self.device=device

        self.attrs = ['u', 'nu', 'ax', 'ay', 'cx', 'cy', 'visc', 'amp'] if('navier_stokes' in path) else \
                     ['u', 'nu', 'ax', 'ay', 'cx', 'cy']
        for attr in self.attrs + ['x', 't']:
            #data_downsampled = np.array(self.downsample(self.data[attr], attr))
            data_downsampled = np.array(self.data[attr])
            setattr(self, attr, 
                    torch.as_tensor(data_downsampled, 
                                    dtype=torch.float32, 
                                    device=device if load_all else 'cpu')
                    )
        f.close()

        if(augmentation is None):
            augmentation = []
        self.augmentation = augmentation
        self.shift = shift
        self.augmentation_ratio = augmentation_ratio
        self.total_samples = len(self.u)
        self.num_samples = num_samples if num_samples > 0 else self.total_samples

        self.dt = self.t[1] - self.t[0]
        # self.x is of shape (2, ny, nx)
        self.dx = self.x[0, 0, 1] - self.x[0, 0, 0]
        self.dy = self.x[1, 1, 0] - self.x[1, 0, 0]

        self.choose_subset(subset, n = self.num_samples)

        print("Data loaded from: {}".format(path))
        print("AUGMENTATION: {}".format(self.augmentation))
        print("num_samples: {}".format(self.num_samples))
        print("Resolution: {}".format(self.u.shape))
        print("Loaded data onto device: {}".format(self.u.device))
        print("\n")

    def __len__(self):
        return self.num_samples*(len(self.augmentation)+1)
    
    def downsample(self, data, attr):
        if attr == 'u':
            return data[:, :self.temporal_horizon, ::self.spatial_downsampling,::self.spatial_downsampling]
        elif attr == 'x':
            return data[:, ::self.spatial_downsampling,::self.spatial_downsampling]
        elif attr == 't':
            return data[:self.temporal_horizon]
        else:
            return data 

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: spatial coordinates
            dict: equation specific parameters
        """
        t_idx = idx % (len(self.augmentation) + 1)
        idx = idx // (len(self.augmentation) + 1)
        idx = self.indexes[idx]
        variables = {attr: getattr(self, attr)[idx] for attr in self.attrs}
        u = variables.pop('u')
        x = self.x
        t = self.t

        #if self.mode == "train" and self.augmentation is not None:
        if self.mode == "train" and self.augmentation != []:
            if self.augmentation_ratio > random.random(): # augment data w/ probability augmentation_ratio
                #pde = self.get_PDE(variables)
                #u = self.augmentation(u, pde, self.shift)
                if(isinstance(self.augmentation[t_idx-1], (Galileo, Scale))):
                    u, x = self.augmentation[t_idx-1]((u,x))
                else:
                    u = self.augmentation[t_idx-1](u)
                    
        return u, x, variables
    
    def get_PDE(self, variables): # pouya didn't change this!
        if variables['ax']!=0 and variables['ay']!=0:
            # advection only has ax and ay
            return "advection"
        elif variables["cx"]!=0 and variables["cy"]!=0:
            # burgers has cx and cy and nu
            return "burgers"
        elif variables["nu"]!=0:
            # heat only has nu
            return "heat"
        else:
            raise ValueError("PDE not found")
        
    def choose_subset(
            self,
            chosen: str = 'heat,adv,burger',
            reverse: bool = False,
            n: int = None,
            ):
        """
        Choose subset of the dataset
        Args:
            chosen: str 
                string of chosen PDEs and subset of PDE coefficients.
                DO NOT USE ANY SPACES!
                Example:
                    'heat,nu>0.5,adv,ax<0.4,burger,cx<0.3'
                Ranges:
                    nu:
                        - burgers: [7.5e-3, 1.5e-2]
                        - heat: [3e-3, 2e-2]
                    ax, ay: [0.1, 2.5]
                    cx, cy: [0.5, 1.0]  
            reverse: bool.
                if True, choose all PDEs except the specified ones
            n: int or None
                number of samples to use from the specified subset
            seed: int
                random seed when choosing n samples (for reproducibility)
        Returns:
            None
        """
        gs = chosen.split(',')

        if 'adv' in gs:
            adv = ((self.ax!=0) | (self.ay!=0)) & ((self.cx==0) & (self.cy==0)) & (self.nu==0)
        else:
            adv = torch.zeros(self.total_samples).bool()

        if 'burger' in gs:
            burger =((self.ax==0) & (self.ay==0)) & ((self.cx!=0) | (self.cy!=0)) & (self.nu!=0)
        else:
            burger = torch.zeros(self.total_samples).bool()

        if 'heat' in gs:
            heat = ((self.ax==0) & (self.ay==0)) & ((self.cx==0) & (self.cy==0)) & (self.nu!=0)
        else:
            heat = torch.zeros(self.total_samples).bool()

        if 'ns' in gs:
            ns = (self.visc != 0) & (self.amp != 0)
        else:
            ns = torch.zeros(self.total_samples).bool()

        for g in gs:
            if '>' in g:
                attr, val = g.split('>')
                if attr in ['ax', 'ay']:
                    adv = adv & (getattr(self, attr)>float(val))
                elif attr in ['cx', 'cy']:
                    burger = burger & (getattr(self, attr)>float(val))
                elif attr in ['nu']:
                    burger = burger & (getattr(self, attr)>float(val))
                    heat = heat & (getattr(self, attr)>float(val))
            elif '<' in g:
                attr, val = g.split('<')
                if attr in ['ax', 'ay']:
                    adv = adv & (getattr(self, attr)<float(val))
                elif attr in ['cx', 'cy']:
                    burger = burger & (getattr(self, attr)<float(val))
                elif attr in ['nu']:
                    burger = burger & (getattr(self, attr)<float(val))
                    heat = heat & (getattr(self, attr)<float(val))
    
        which = heat.to(self.device) | adv.to(self.device) | burger.to(self.device) | ns.to(self.device)
        if reverse:
            which = ~which

        self.indexes = torch.arange(self.total_samples, device=which.device)[which]

        if type(n) is int:
            if n > len(self.indexes):
                print(f"You want {n} samples but there are only {len(self.indexes)} available. Overriding {n} to {len(self.indexes)}")
                self.num_samples = len(self.indexes)
                n = len(self.indexes)

            self.indexes = self.indexes[np.random.choice(len(self.indexes), n, replace=False)]
        
        # Check number of equations
        eq_dict = {"heat": 0, "adv": 0, "burgers": 0, "ns": 0}
        for idx in self.indexes:
            eq = self.get_eq(idx)
            eq_dict[eq] += 1

        print(eq_dict)
    
    def cleanup(self):
        del self.u
        del self.x
        del self.t
        del self.nu
        del self.ax
        del self.ay
        del self.cx
        del self.cy
    
        
    def get_eq(self, idx):
        nu = self.nu[idx]
        cx = self.cx[idx]
        amp = self.amp[idx] if 'amp' in self.attrs else 0

        if amp != 0:
            return "ns"
        if nu == 0:
            return "adv"
        if cx == 0:
            return "heat"
        else:
            return "burgers"
