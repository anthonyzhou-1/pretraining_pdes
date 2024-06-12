import torch
#import deepxde as dde
#from deepxde.nn.pytorch.fnn import FNN
#from deepxde.nn.pytorch.nn import NN
#from deepxde.nn import activations

from torch import nn
from torch.nn import functional as F
import copy 


#class DeepONet1D(NN):
class DeepONet1D(nn.Module):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        time_window,
        time_future,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        eq_variables,
        regularization=None,
        seed=None,
    ):
        super().__init__()
        if(seed is not None):
            torch.manual_seed(seed)
        self.activation = nn.ReLU if(activation == 'relu') else nn.SiLU() if(activation == "silu") else nn.Tanh()

        # Add input and output sizes to networks
        layer_sizes_branch.insert(0, len(eq_variables)+2+time_window)
        layer_sizes_branch.append(time_future)
        layer_sizes_trunk.insert(0, 1)
        layer_sizes_trunk.append(time_future)

        # Get branch
        branch = []
        for i in range(len(layer_sizes_branch)-1):
            branch.append(nn.Linear(layer_sizes_branch[i], layer_sizes_branch[i+1]))
            branch.append(self.activation)
        self.branch = nn.Sequential(*branch[:-1])

        # Get trunk
        trunk = []
        for i in range(len(layer_sizes_trunk)-1):
            trunk.append(nn.Linear(layer_sizes_trunk[i], layer_sizes_trunk[i+1]))
            trunk.append(self.activation)
        self.trunk = nn.Sequential(*trunk[:-1])

        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.eq_variables = eq_variables

    def forward(self,
                u: torch.Tensor,
                grid: torch.Tensor,
                dt: torch.Tensor,
                variables: dict = None,) -> torch.Tensor:

        u = u.permute(0, 2, 1) # (batch, time_history, x) -> (batch, x, time_history)
        b, nx, _ = u.shape

        # Process input
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

        # Branch net to encode the input function
        x_func = self.branch(x) 
        # Trunk net to encode the domain of the output function
        #if self._input_transform is not None:
        #    x_loc = self._input_transform(x_loc)

        #x_loc = self.activation_trunk(self.trunk(grid))
        if(len(grid.shape) == 2):
            grid = grid.unsqueeze(-1)
        x_loc = self.trunk(grid)

        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        #print()
        #print()
        #print(x_func.shape, x_loc.shape)
        x = torch.einsum("mbi,mbi->mb", x_func, x_loc)
        # Add bias
        x += self.b

        #if self._output_transform is not None:
        #    x = self._output_transform(inputs, x)
        #print(x.shape)
        #print()
        #print()
        #raise
        return x.unsqueeze(1)

    def pretrain(self):
        self._pretrain = True

    def pretrain_off(self):
        self._pretrain = False


class DeepONet2D(nn.Module):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        time_window,
        time_future,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        eq_variables,
        regularization=None,
        seed=None,
    ):
        super().__init__()
        if(seed is not None):
            torch.manual_seed(seed)
        self.activation = nn.ReLU if(activation == 'relu') else nn.SiLU() if(activation == "silu") else nn.Tanh()

        self.time_future = time_future
        # Add input and output sizes to networks
        self.layer_sizes_branch = copy.deepcopy(layer_sizes_branch)
        self.layer_sizes_branch.insert(0, len(eq_variables)+3+time_window)
        self.layer_sizes_branch.append(time_future)

        self.layer_sizes_trunk = copy.deepcopy(layer_sizes_trunk)
        self.layer_sizes_trunk.insert(0, 2)
        self.layer_sizes_trunk.append(time_future)

        # Get branch
        branch = []
        for i in range(len(self.layer_sizes_branch)-1):
            branch.append(nn.Linear(self.layer_sizes_branch[i], self.layer_sizes_branch[i+1]))
            branch.append(self.activation)
        self.branch = nn.Sequential(*branch[:-1])

        # Get trunk
        trunk = []
        for i in range(len(self.layer_sizes_trunk)-1):
            trunk.append(nn.Linear(self.layer_sizes_trunk[i], self.layer_sizes_trunk[i+1]))
            trunk.append(self.activation)
        self.trunk = nn.Sequential(*trunk[:-1])

        self.b = torch.nn.parameter.Parameter(torch.zeros(time_future))
        self.regularizer = regularization
        self.eq_variables = eq_variables

    def forward(self,
                u: torch.Tensor,
                grid: torch.Tensor,
                dt: torch.Tensor,
                variables: dict = None,
                embeddings:torch.Tensor = None) -> torch.Tensor:

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
        if embeddings is not None:
            x = torch.cat((x, embeddings.unsqueeze(1).unsqueeze(2).repeat(1, nx, ny, 1).to(u.device)), -1) # embeddings in shape (batch, embedding_dim)
                                    
        # Branch net to encode the input function
        grid = grid.permute(0, 2, 3, 1) # (batch, time_history, x, y) -> (batch, x, y, time_history)
        x_func = self.branch(x) # (batch, x, y, time_history) -> (batch, x, y, time_future)

        if(len(grid.shape) == 2):
            grid = grid.unsqueeze(-1)
        x_loc = self.trunk(grid) # (batch, x, y, 2) -> (batch, x, y, time_future)

        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )

        if self.time_future > 1:
            # using split_both strategy from deepxde
            # x_func, x_loc is in shape (batch, x, y, time_future)
            x = torch.zeros(b, nx, ny, self.time_future, device=u.device)
            for i in range(self.time_future):
                x_func_i = x_func[..., i].unsqueeze(-1)
                x_loc_i = x_loc[..., i].unsqueeze(-1)
                x_i = torch.einsum("mnbi,mnbi->mnb", x_func_i, x_loc_i)
                x[..., i] = x_i
                
        else:
            x = torch.einsum("mnbi,mnbi->mnb", x_func, x_loc)

        # Add bias
        x += self.b

        return x.unsqueeze(1) if self.time_future == 1 else x.permute(0, 3, 1, 2)

    def __repr__(self):
        return "DeepONet2D"

    def pretrain(self):
        self._pretrain = True

    def pretrain_off(self):
        self._pretrain = False
