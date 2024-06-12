
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d_bundled(nn.Module):
    def __init__(self,
                time_window: int = 20,
                time_future: int = 1,
                modes1: int = 6,
                modes2: int = 6,
                width: int = 24,
                num_layers: int = 4,
                eq_variables: dict = {},
                dropout = 0.1):
        super(FNO2d_bundled, self).__init__()
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
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.eq_variables = eq_variables   
        self.in_channels = time_window  + 3 + len(eq_variables) # Time window + dx + dy + dt + embeddings + vars
        self.out_channels = time_future
        self.fc0 = nn.Linear(self.in_channels, self.width) 
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)
        self.dropout = nn.Dropout(dropout)

        fourier_layers = []
        conv_layers = []
        for i in range(num_layers):
            fourier_layers.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            conv_layers.append(nn.Conv2d(self.width, self.width, 1))
        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.conv_layers = nn.ModuleList(conv_layers)

    def __repr__(self):
        return f'FNO2d'

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
        u = u.permute(0, 2, 3, 1) # (batch, time_history, x, y) -> (batch, x, y, time_history)
        b = u.shape[0]
        nx = u.shape[1]
        ny = u.shape[2]

        dx = grid[:, 0, 0, 1] - grid[:, 0, 0, 0]
        dy = grid[:, 1, 1, 0] - grid[:, 1, 0, 0]

        x = torch.cat((u, dx[:, None, None, None].to(u.device).repeat(1, nx, ny, 1)), -1)
        x = torch.cat((x, dy[:, None, None, None].to(u.device).repeat(1, nx, ny, 1)), -1)
        x = torch.cat((x, dt[:, None, None, None].to(u.device).repeat(b, nx, ny, 1)), -1)
        # x is in shape (batch, x, y, time_history + 3)
        if "nu" in self.eq_variables.keys():
            nu = variables["nu"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["nu"]
            x = torch.cat((x, nu.to(u.device)), -1)
        if "ax" in self.eq_variables.keys():
            ax = variables["ax"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["ax"]
            x = torch.cat((x, ax.to(u.device)), -1)
        if "ay" in self.eq_variables.keys():
            ay = variables["ay"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["ay"]
            x = torch.cat((x, ay.to(u.device)), -1)
        if "cx" in self.eq_variables.keys():
            cx = variables["cx"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["cx"]
            x = torch.cat((x, cx.to(u.device)), -1)
        if "cy" in self.eq_variables.keys():
            cy = variables["cy"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["cy"]
            x = torch.cat((x, cy.to(u.device)), -1)
       
        x = self.fc0(x) # (batch, x, y, channel -> batch, x, y, width)
        
        # [b, x, y, c] -> [b, c, x, y]
        x = x.permute(0, 3, 1, 2)

        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)
            x = self.dropout(x)

        # [b, c, x, y] -> [b, x, y, c]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # [b, x, y, t] -> [b, t, x, y]
        x = x.permute(0, 3, 1, 2)

        return x

class FNO_pretrain(nn.Module):
    def __init__(self, modes, tw_in, width, h_size):
        super(FNO_pretrain, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes
        self.modes2 = modes
        self.tw_in = tw_in
        self.width = width
        self.h_size = h_size

        self.fc0 = nn.Linear(self.tw_in+2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y, t)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 4, 2)
        self.w3 = nn.Conv2d(self.width, self.width, 4, 2)

        #self.fc1 = nn.Linear(432, 128)
        #self.fc1 = nn.Linear(self.width, 128)
        self.fc1 = nn.Linear(1728, 128) # TODO: Make this not hardcoded?
        self.fc2 = nn.Linear(128, self.h_size)

    def forward(self, x, grid):
        x = x.permute(0, 2, 3, 1)
        grid = grid.permute(0, 2, 3, 1)
        #print(x.shape)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        #print(x.shape)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        #print(x.shape)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        #print(x.shape)

        x = self.w2(x)
        x = F.gelu(x)
        #print(x.shape)

        x = self.w3(x)
        x = F.gelu(x)
        #print(x.shape)

        x = x.reshape(x.shape[0], -1)
        #print(x.shape)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class SpectralConv2d_dy(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, K):
        super(SpectralConv2d_dy, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.K = K

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(K, in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(K, in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights, att):
        # input: (batch, in_channel, x,y )
        # weights: (K, in_channel, out_channel, x,y)
        # att: (batch, K)
        # output: (batch, out_channel, x,y)
        weights = torch.einsum("bk, kioxy->bioxy", att, weights)
        return torch.einsum("bixy,bioxy->boxy", input, weights)

    def forward(self, x, att):
        batchsize = x.shape[0]
        att = torch.complex(att, torch.zeros_like(att))
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1, att)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2, att)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d_piano(nn.Module):
    def __init__(self,
                time_window: int = 20,
                time_future: int = 1,
                modes1: int = 6,
                modes2: int = 6,
                width: int = 24,
                num_layers: int = 4,
                eq_variables: dict = {},
                seed = None):
        super(FNO2d_bundled, self).__init__()
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

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.eq_variables = eq_variables   
        self.in_channels = time_window  + 3 + len(eq_variables) # Time window + dx + dy + dt + embeddings + vars
        self.out_channels = time_future
        self.fc0 = nn.Linear(self.in_channels, self.width) 
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

        fourier_layers = []
        conv_layers = []
        for i in range(num_layers):
            fourier_layers.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2, seed=seed))
            conv_layers.append(nn.Conv2d(self.width, self.width, 1))
        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.conv_layers = nn.ModuleList(conv_layers)

    def __repr__(self):
        return f'FNO2d'

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
        u = u.permute(0, 2, 3, 1) # (batch, time_history, x, y) -> (batch, x, y, time_history)
        b = u.shape[0]
        nx = u.shape[1]
        ny = u.shape[2]

        dx = grid[:, 0, 0, 1] - grid[:, 0, 0, 0]
        dy = grid[:, 1, 1, 0] - grid[:, 1, 0, 0]

        x = torch.cat((u, dx[:, None, None, None].to(u.device).repeat(1, nx, ny, 1)), -1)
        x = torch.cat((x, dy[:, None, None, None].to(u.device).repeat(1, nx, ny, 1)), -1)
        x = torch.cat((x, dt[:, None, None, None].to(u.device).repeat(b, nx, ny, 1)), -1)
        # x is in shape (batch, x, y, time_history + 3)
        if "nu" in self.eq_variables.keys():
            nu = variables["nu"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["nu"]
            x = torch.cat((x, nu.to(u.device)), -1)
        if "ax" in self.eq_variables.keys():
            ax = variables["ax"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["ax"]
            x = torch.cat((x, ax.to(u.device)), -1)
        if "ay" in self.eq_variables.keys():
            ay = variables["ay"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["ay"]
            x = torch.cat((x, ay.to(u.device)), -1)
        if "cx" in self.eq_variables.keys():
            cx = variables["cx"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["cx"]
            x = torch.cat((x, cx.to(u.device)), -1)
        if "cy" in self.eq_variables.keys():
            cy = variables["cy"][:, None, None, None].repeat(1, nx, ny, 1) / self.eq_variables["cy"]
            x = torch.cat((x, cy.to(u.device)), -1)
       
        x = self.fc0(x) # (batch, x, y, channel -> batch, x, y, width)
        
        # [b, x, y, c] -> [b, c, x, y]
        x = x.permute(0, 3, 1, 2)

        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)
            x = self.dropout(x)

        # [b, c, x, y] -> [b, x, y, c]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # [b, x, y, t] -> [b, t, x, y]
        x = x.permute(0, 3, 1, 2)

        return x


class PIANO_FNO(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=21, d_embds=256, K=4, T=10.0):
        super(PIANO_FNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.d_embds = d_embds
        self.K = K
        self.T = T
        self.padding = 4

        self.fc0 = nn.Linear(10, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y, t)

        self.conv0 = SpectralConv2d_dy(self.width, self.width, self.modes1, self.modes2, self.K)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.attfc0 = nn.Sequential(nn.Linear(self.d_embds, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, K))

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, grid, embedding):
        #grid = self.get_grid(x.shape, x.device)
        x = x.permute(0, 2, 3, 1)
        grid = grid.permute(0, 2, 3, 1)

        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        att = self.attfc0(embedding)
        att = F.softmax(att/self.T, dim=-1)
        x1 = self.conv0(x, att)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1 - 1 / size_x, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1 - 1 / size_y, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

