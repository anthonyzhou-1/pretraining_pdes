"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

#Complex multiplication

##########################################################s######
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d_bundled(nn.Module):
    def __init__(self,
                time_window: int = 25,
                time_future: int = 1,
                modes: int = 32,
                width: int = 256,
                num_layers: int = 5,
                eq_variables: dict = {},
                seed=None):
        super(FNO1d_bundled, self).__init__()
        """
        Initialize the overall FNO network. It contains 5 layers of the Fourier layer.
        The input to the forward pass has the shape [batch, time_history, x].
        The output has the shape [batch, time_future, x].
        Args:
            time_history (int): input timesteps of the trajectory
            time_future (int): output timesteps of the trajectory
            modes (int): low frequency Fourier modes considered for multiplication in the Fourier space
            width (int): hidden channel dimension
            num_layers (int): number of FNO layers
        """
        if(seed is not None):
            torch.manual_seed(seed)
        self.modes = modes
        self.width = width
        self.eq_variables = eq_variables   
        self.in_channels = time_window + 2 + len(eq_variables) # Time window + dx + dt + vars
        self.out_channels = time_future
        self.fc0 = nn.Linear(self.in_channels, self.width) 
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

        fourier_layers = []
        conv_layers = []
        for i in range(num_layers):
            fourier_layers.append(SpectralConv1d(self.width, self.width, self.modes))
            conv_layers.append(nn.Conv1d(self.width, self.width, 1))
        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.conv_layers = nn.ModuleList(conv_layers)

    def __repr__(self):
        return f'FNO1d'

    def forward(self, 
                u: torch.Tensor, 
                grid: torch.Tensor,
                dt: torch.Tensor, 
                variables: dict = None,) -> torch.Tensor:
        """
        Forward pass of FNO network.
        The input to the forward pass has the shape [batch, time_history, x].
        1. Add dx and dt as channel dimension to the time_history, repeat for every x
        2. Lift the input to the desired channel dimension by self.fc0
        3. 5 (default) FNO layers
        4. Project from the channel space to the output space by self.fc1 and self.fc2.
        The output has the shape [batch, time_future, x].
        Args:
            u (torch.Tensor): input tensor of shape [batch, time_history, x]
            dx (torch.Tensor): spatial distances
            dt (torch.Tensor): temporal distances
        Returns: torch.Tensor: output has the shape [batch, time_future, x]
        """
        # TODO: rewrite training method and forward pass without permutation
        #print("Us SHAPES:")
        #print(u.shape)
        u = u.permute(0, 2, 1) # (batch, time_history, x) -> (batch, x, time_history)
        b, nx, _ = u.shape

        #print("GRID SHAPE: {}".format(grid.shape))
        dx = grid[:, 1] - grid[:, 0] # (batch)

        x = torch.cat((u, dx[..., None, None].repeat(1, nx, 1).to(u.device)), -1) # x_pos is in shape (batch, x)
        x = torch.cat((x, dt[:, None, None].repeat(b, nx, 1).to(u.device)), -1) # t_pos in shape (batch)
        if "alpha" in self.eq_variables.keys():
            alpha = variables["alpha"][:, None, None].repeat(1, nx, 1) / self.eq_variables["alpha"]
            x = torch.cat((x, alpha.to(u.device)), -1)
        if "beta" in self.eq_variables.keys():
            beta = variables["beta"][:, None, None].repeat(1, nx, 1) / self.eq_variables["beta"]
            x = torch.cat((x, beta.to(u.device)), -1)
        if "gamma" in self.eq_variables.keys():
            gamma = variables["gamma"][:, None, None].repeat(1, nx, 1) / self.eq_variables["gamma"]
            x = torch.cat((x, gamma.to(u.device)), -1)
     
        #print("X SHAPE AGAIN: {}".format(x.shape))
        #print(self.fc0)
        #print()
        #raise
        x = self.fc0(x) # (batch, x, channel -> batch, x, width)
        # [b, x, c] -> [b, c, x]
        x = x.permute(0, 2, 1)

        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)

        # [b, c, x] -> [b, x, c]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        #print("\nFINAL X SHAPE: {}\n".format(x.shape))

        return x.permute(0, 2, 1)

