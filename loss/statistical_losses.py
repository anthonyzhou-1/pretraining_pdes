import torch
import torch.nn.functional as F
import torch.nn as nn
from common.utils import dict2tensor
from typing import Tuple
from findiff import FinDiff
from loss.loss_models import CNN_Decoder, CNN_classifier
from common.utils import LpLoss
from einops.layers.torch import Rearrange

################################################################
# Loss Functions
################################################################
class Masked_Regression(nn.Module):
    def __init__(self, args, model):
        super(Masked_Regression, self).__init__()
        self.args = args
        self.encoder = model
        self.spatial_patch_size = args.spatial_patch_size
        self.temporal_patch_size = args.temporal_patch_size
        self.mask_ratio = args.mask_ratio
        self.device=args.device
        self.patch_dim = self.temporal_patch_size * self.spatial_patch_size * self.spatial_patch_size
        self.criterion = nn.L1Loss(reduction="none")

        self.mask_token = nn.Parameter(torch.zeros(1, self.patch_dim))
        
        self.patchify = Rearrange('b (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf)',
                                  pf=self.temporal_patch_size,
                                  p1=self.spatial_patch_size,
                                  p2=self.spatial_patch_size,)
        
        self.unpatchify = Rearrange('b (f h w) (p1 p2 pf) -> b (f pf) (h p1) (w p2)',
                                    pf=self.temporal_patch_size,
                                    p1=self.spatial_patch_size,
                                    p2=self.spatial_patch_size,
                                    f = args.time_window//self.temporal_patch_size,
                                    h = args.base_resolution[1]//self.spatial_patch_size,
                                    w = args.base_resolution[2]//self.spatial_patch_size)
        
        self.decoder = CNN_Decoder(in_channels=args.time_future, out_channels=args.time_window)

    def forward(self, u, grid, dt, variables, testing=False):
        u_masked, mask = self.get_masked_input(u)
        
        z = self.encoder(u_masked, grid, dt, variables) # [b, 1, nx, ny]
        u_rec = self.decoder(z) # [b, tw, nx, ny]

        loss = self.criterion(u_rec, u)
        loss = (loss * mask).sum() / (mask.sum() + 1e-5)

        if testing:
            return loss, u, u_masked, u_rec

        return loss
    
    def get_masked_input(self, u):
        u_patch = self.patchify(u) # [b, f h w, p1 p2 pf]
        num_patches = u_patch.shape[1]
        batch = u_patch.shape[0]

        num_masked = int(self.mask_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = self.device).argsort(dim = -1)
        masked_indices = rand_indices[:, :num_masked]

        batch_range = torch.arange(batch, device = self.device)[:, None]
        u_patch[batch_range, masked_indices] = self.mask_token

        u_out = self.unpatchify(u_patch) # [b, f pf, h, w]

        mask = torch.zeros(batch, num_patches, self.patch_dim, device=self.device)
        mask[batch_range, masked_indices] = 1
        mask = self.unpatchify(mask) # [b, f pf, h, w]

        return u_out, mask


class Derivative_Regression(nn.Module):
    def __init__(self, args, model, d_x, d_y, d_t):
        super(Derivative_Regression, self).__init__()
        self.args = args
        self.encoder = model
        self.dx = FinDiff(2, d_x)
        self.dy = FinDiff(3, d_y)
        self.dxx = FinDiff(2, d_x, 2)
        self.dyy = FinDiff(3, d_y, 2)
        self.dt = FinDiff(1, d_t)
        self.decoder = CNN_Decoder(in_channels=args.time_future,
                                   hidden_size=args.CNN_decoder_hidden_dim)
        self.criterion = LpLoss(2, 2)

    def forward(self, u, grid, dt, variables, testing=False):
        ut = self.dt(u)[:, 0] # Take the derivative of the first frame only
        ux = self.dx(u)[:, 0]
        uy = self.dy(u)[:, 0]
        uxx = self.dxx(u)[:, 0]
        uyy = self.dyy(u)[:, 0]

        derivatives = torch.stack([ut, ux, uy, uxx, uyy], dim=1) # [b, 5, nx, ny]

        z = self.encoder(u, grid, dt, variables) # [b, 1, nx, ny]
        y = self.decoder(z) # [b, 5, nx, ny]

        loss = self.criterion(y, derivatives)

        if testing:
            return loss, y, derivatives, u

        return loss

class Coefficient_Regression(nn.Module):
    def __init__(self, args, model):
        super(Coefficient_Regression, self).__init__()
        self.args = args
        self.encoder = model
        self.pde = args.pde
        self.decoder = CNN_classifier(args)

    def forward(self, u, grid, dt, variables, testing=False):
        coeffs = norm_coeffs(variables, self.pde)
        coeffs = dict2tensor(coeffs).to(u.device)

        z = self.encoder(u, grid, dt, variables) # [b, 1, nx, ny]
        y = self.decoder(z) # [b, n_classes]
        loss = F.mse_loss(y, coeffs)

        if testing:
            return loss, y, coeffs

        return loss


################################################################
# Utils
################################################################

def norm_coeffs(coeffs, pde):
    """
    Normalize coefficients
    Args:
        coeffs (torch.Tensor): coefficients
        pde (str): PDE (e.g. 'heat_adv_burgers')
    Returns:
        torch.Tensor: normalized coefficients
    """

    eq_variables = {}
    if pde == "kdv_burgers":
        eq_variables['alpha'] = 3.
        eq_variables['beta'] = 0.4
        eq_variables['gamma'] = 1.
    elif pde == "heat_adv_burgers":
        # Set maximum values for the parameters to normalize them in model
        eq_variables['nu'] = 2e-2
        eq_variables['ax'] = 2.5
        eq_variables['ay'] = 2.5
        eq_variables['cx'] = 1.0
        eq_variables['cy'] = 1.0

    for key in eq_variables.keys():
        coeffs[key] = coeffs[key] / eq_variables[key]
    
    return coeffs