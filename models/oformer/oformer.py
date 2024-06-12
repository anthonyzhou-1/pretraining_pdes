import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from .attention_module import pair, PreNorm, PostNorm,\
    StandardAttention, FeedForward, LinearAttention, ReLUFeedForward
from .cnn_module import PeriodicConv2d, PeriodicConv3d, UpBlock

from .encoder_module import Encoder1D
from .decoder_module import PointWiseDecoder1D


class OFormer1D(nn.Module):
    def __init__(self, encoder, decoder, time_future, eq_variables={}):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.time_future = time_future
        self._pretrain = False
        self._finetune = False
        self.eq_variables = eq_variables

        #self.finetune = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256,256), nn.SiLU(), nn.Linear(256,20))
        #self.finetune = nn.Sequential(nn.Linear(128, 10))

    #def forward(self, x, input_pos):
    def forward(self, 
                u: torch.Tensor, 
                grid: torch.Tensor,
                dt: torch.Tensor, 
                variables: dict = None, 
                embeddings:torch.Tensor = None) -> torch.Tensor:

        u = u.permute(0, 2, 1) # (batch, time_history, x) -> (batch, x, time_history)
        b, nx, _ = u.shape

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

        #print(x.shape, grid.shape)
        
        grid = grid.unsqueeze(-1)
        h = self.encoder(x, grid)
        #if(self._pretrain):
        #    #print(h.shape)
        #    out = self.finetune(h)
        #    #print(out.shape)
        #    return out
        #    raise
        #    return h

        #out = self.decoder.rollout(h, propagate_pos=grid, input_pos=grid, forward_steps=self.time_future)
        out = self.decoder(h, propagate_pos=grid, input_pos=grid)
        #if(self._pretrain):
        #    return out

        #print(out.shape)
        #print(self._finetune)
        #if(self._finetune):
            #print("\nFINETUNEING\n")
        #    out = self.finetune(out)
        #print(out.shape)
        #raise
        return out.transpose(1,2)

    def get_loss(self, x, y, input_pos, loss_fn):
        y_pred = self.forward(x, input_pos)[...,0]
        return y_pred, loss_fn(y_pred, y.unsqueeze(-1))

    def pretrain(self):
        self._pretrain = True
        self.decoder.pretrain()

    def pretrain_off(self):
        self._pretrain = False
        self.decoder.pretrain_off()


class OFormer2D(nn.Module):
    def __init__(self, encoder, decoder, time_future, eq_variables):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.time_future = time_future
        self._pretrain = False
        self._finetune = False
        self.eq_variables = eq_variables

    #def forward(self, x, input_pos):
    def forward(self, 
                u: torch.Tensor, 
                grid: torch.Tensor,
                dt: torch.Tensor, 
                variables: dict = None,) -> torch.Tensor:

        u = u.permute(0, 2, 3, 1) # (batch, time_history, x, y) -> (batch, x, y, time_history)
        b = u.shape[0]
        nx = u.shape[1]
        ny = u.shape[2]
        #print("SHOULD BE 16, 64, 64, 20: {}".format(u.shape))

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


        x = x.flatten(1,2)
        grid = grid.permute(0, 2, 3, 1) # (batch, time_history, x, y) -> (batch, x, y, time_history)
        grid = grid.flatten(1,2)
        h = self.encoder(x, grid)
        out, _ = self.decoder(h, propagate_pos=grid, input_pos=grid)
        return out.reshape(out.shape[0], out.shape[1], nx, ny)
    
    def __repr__(self):
        return f'OFormer2D'

    def get_loss(self, x, y, input_pos, loss_fn):
        y_pred = self.forward(x, input_pos)
        return y_pred, loss_fn(y_pred, y)

    def ssl(self):
        self._ssl = True


