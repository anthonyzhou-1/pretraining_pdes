import torch
import numpy as np
import random
import torch.nn.functional as F
import itertools
import math
import scipy 
from einops import rearrange
from functools import lru_cache
import torch.nn as nn
from loss.loss_models import *

################################################################
# Loss Functions
################################################################
class Binary_Sort(nn.Module):
    def __init__(self, config, model):
        super(Binary_Sort, self).__init__()
        self.config = config
        self.encoder = model
        self.decoder = get_decoder(config)
        self.weighted_sample = config.weighted_sample

    def forward(self, u, grid, dt, variables, difficulty=0, testing=False):
        samples, labels, pos_weight = shuffle_odd(u, difficulty, self.weighted_sample)

        z = self.encoder(samples, grid, dt, variables)
        logits = self.decoder(z).squeeze()

        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        if testing:
            return loss, samples, logits, labels

        return loss
    
class Sort(nn.Module):
    def __init__(self, config, model):
        super(Sort, self).__init__()
        self.config = config
        self.encoder = model
        self.decoder = get_decoder(config)
        self.mode = "spatial" if config.pretraining == "sort_space" else "temporal"
        self.patch_size = config.spatial_patch_size if self.mode == "spatial" else config.temporal_patch_size

    def forward(self, u, grid, dt, variables, testing=False):
        '''
        SequenceSorting loss function. https://arxiv.org/pdf/1708.01246.pdf
        Multi-way classification to detect the order of a shuffled sample.
        Expects model to output a tensor of shape [batch, n_classes] without softmax
        We set patch_size to bundle time or spatial data together
            - Reasoning is that consecutive timesteps may not differ much, so we shuffle patches
        Note: This can be viewed as a specific case of jigsaw, where patches/puzzle pieces are only shuffled temporally or spatially. 
        Args:
            u (torch.Tensor): input tensor of shape [batch, time_history, x, (y)]
            grid: grid
            dt: time step
            variables: variables
            model (nn.Module): model to use
            patch_size (int, optional): size of temporal patch. Defaults to 2.
            mode (str, optional): mode of shuffling. Defaults to "temporal".
        '''
        
        u_shuffled, labels = shuffle_samples(u, self.patch_size, mode=self.mode)
        z = self.encoder(u_shuffled, grid, dt, variables)
        logits = self.decoder(z)

        loss = F.cross_entropy(logits, labels)

        if testing:
            return loss, u_shuffled, logits, labels

        return loss

class Jigsaw(nn.Module):
    def __init__(self, config, model):
        super(Jigsaw, self).__init__()
        self.config = config
        self.encoder = model
        self.decoder = get_decoder(config)
        self.spatial_patch_size = config.spatial_patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.n_classes = config.n_classes

    def forward(self, u, grid, dt, variables, testing=False):
        '''
        Jigsaw loss function. https://arxiv.org/pdf/1603.09246.pdf
        Multi-way classification to detect the order of a shuffled sample.
        Expects model to output a tensor of shape [batch, n_classes] without softmax
        We shuffle across both time and space, and predict the label.
        This introduces a large number of permutations, so we cap the number of classes
            - Number of permutations: [(nx/px) * (ny/py) * (time_history/pt)]! 
            - We choose the largest Hamming distance between the shuffled and unshuffled samples
        Args:
            u (torch.Tensor): input tensor of shape [batch, time_history, x, (y)]
            model (nn.Module): model to use 
            spatial_patch_size (int): size of spatial patch
            temporal_patch_size (int): size of temporal patch
            n_classes (int, optional): number of classes. Defaults to 50.
        '''

        u_shuffled, labels = shuffle_jigsaw(u, self.temporal_patch_size, self.spatial_patch_size, num_classes=self.n_classes)
        z = self.encoder(u_shuffled, grid, dt, variables)
        logits = self.decoder(z)

        loss = F.cross_entropy(logits, labels)

        if testing:
            return loss, u_shuffled, logits, labels

        return loss

################################################################
# Utils
################################################################
def get_decoder(config):
    if config.projector == 'MLP':
        decoder = MLP_classifier(config)
    elif config.projector == 'CNN':
        decoder = CNN_classifier(config)
    else:
        raise NotImplementedError

    return decoder

def shuffle_odd(u, difficulty: float = 0., weighted_sample: float = 0.5):
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0
    samples = torch.zeros_like(u, device=u.device)
    batch_size = u.shape[0]
    labels = torch.zeros(batch_size, device=u.device)
    pos_weight = torch.zeros(batch_size, device=u.device)

    for b in range(batch_size):
        if random.random() < weighted_sample:
            samples[b] = get_negative_sample(u[b], difficulty)
            labels[b] = NEGATIVE_LABEL
            pos_weight[b] = 1 - weighted_sample
        else:
            samples[b] = u[b]
            labels[b] = POSITIVE_LABEL
            pos_weight[b] = weighted_sample

    return samples, labels, pos_weight

def get_negative_sample(u, difficulty: float =0.):
        """
        Get a negative sample from the dataset.
        We shuffle a subset of the input according to the difficulty.
        Args:
            u (torch.Tensor): input tensor of shape [time_history, x, (y)]
            difficulty (float, optional): difficulty level of the negative sample. Defaults to 1.
                0. = all samples are shuffled in the negative sample
                1. = no samples are shuffled in the negative sample
                Reasoning is that if all samples are shuffled in a negative sample it is easier to distinguish from unshuffled samples.
                We can use this to gradually increase difficulty during pretraining
        Returns:
            torch.Tensor: negative sample of shape [batch, time_history, x, (y)]
        """
        time_history = u.shape[0]
        output = torch.clone(u)
        num_shuffle = int(time_history * (1-difficulty)) + 1
        if num_shuffle < 2:
            num_shuffle = 2 # Ensure that at least 2 samples are shuffled
        elif num_shuffle > time_history:
            num_shuffle = time_history 

        indices = random.sample(list(range(time_history)), num_shuffle)
        shuffled_subset = shuffle(u[indices])
        output[indices] = shuffled_subset
        assert torch.all(output == u) == False, "Negative sample is the same as the input"
        return output

def patchify(u, patch_size, mode="temporal", jigsaw_patch_size=0):
    batch_size, time_history, x = u.shape[:3]
    if mode == "temporal":
        num_patches = time_history // patch_size
        if len(u.shape) == 3:
            u = rearrange(u, 'b (n p) x -> b n p x', p = patch_size, n = num_patches)
        else:
            u = rearrange(u, 'b (n p) x y -> b n p x y', p = patch_size, n = num_patches)
    
    elif mode == "spatial":
        num_patches = x // patch_size
        if len(u.shape) == 3:
            u = rearrange(u, 'b t (n p) -> b n p t', p=patch_size, n=num_patches)
        else:
            u = rearrange(u, 'b t (nx px) (ny py) -> b (nx ny) px py t', px=patch_size, py=patch_size)
            num_patches = num_patches ** 2
    
    elif mode == "jigsaw":
        temporal_patch_size = patch_size
        spatial_patch_size = jigsaw_patch_size
        if (len(u.shape) == 3):
            num_patches = (time_history // temporal_patch_size) * (x // spatial_patch_size)
            u = rearrange(u, 'b (nt pt) (nx px) -> b (nt nx) pt px', pt=temporal_patch_size, px=spatial_patch_size)
        else:
            num_patches = (time_history // temporal_patch_size) * (x // spatial_patch_size) ** 2
            u = rearrange(u, 'b (nt pt) (nx px) (ny py) -> b (nt nx ny) pt px py', pt=temporal_patch_size, px=spatial_patch_size, py=spatial_patch_size)

    else:
        raise ValueError("Invalid mode. Choose 'temporal', 'spatial' or 'jigsaw'")
    
    return u, num_patches
    
def unpatchify(u, patch_size, num_patches, mode="temporal", jigsaw_img_size=0):
    if mode == "temporal":
        if len(u.shape) == 4:
            u = rearrange(u, 'b n p x -> b (n p) x', p = patch_size, n = num_patches)
        else:
            u = rearrange(u, 'b n p x y -> b (n p) x y', p = patch_size, n = num_patches)
    elif mode == "spatial":
        if len(u.shape) == 4:
            u = rearrange(u, 'b n p t -> b t (n p)', p = patch_size, n = num_patches)
        else:
            num_patches = int(np.sqrt(num_patches))
            u = rearrange(u, 'b (nx ny) px py t -> b t (nx px) (ny py)', px=patch_size, py=patch_size, nx=num_patches)
    elif mode == "jigsaw":
        if len(u.shape) == 4:
            nt = jigsaw_img_size // patch_size
            u = rearrange(u, 'b (nt nx) pt px -> b (nt pt) (nx px)', nt = nt)
        else:
            nt = jigsaw_img_size // patch_size
            nx = int(np.sqrt(num_patches//nt))
            u = rearrange(u, 'b (nt nx ny) pt px py -> b (nt pt) (nx px) (ny py)', nt=nt, nx=nx)
    else:
        raise ValueError("Invalid mode. Choose 'temporal' or 'spatial'")
    return u


def shuffle_samples(u, patch_size=2, mode="temporal"):
    """
    Shuffle samples in a sequence.
    Args:
        u (torch.Tensor): input tensor of shape [batch, time_history, x, (y)]
        patch_size (int, optional): size of the tuple to sort. Defaults to 2.
        num_classes (int, optional): number of classes. Defaults to 24.
        mode (str, optional): mode of shuffling. Defaults to "temporal".
            - temporal: shuffle the time dimension
            - spatial: shuffle the spatial dimension
    Returns:
        torch.Tensor: shuffled input tensor of shape [batch, time_history, x, (y)]
        torch.Tensor: labels of shape [batch]
    """
    batch_size = u.shape[0]
    u_patch, num_patches = patchify(u, patch_size, mode)

    all_permutations = get_permutations(num_patches).to(u.device)
    batch_range = torch.arange(batch_size, device = u.device)[:, None]

    # Get (batch_size) number of permutations
    random_selection = []
    random_idx = []
    for idx in np.random.randint(0, len(all_permutations), batch_size):
        random_selection.append(all_permutations[idx])
        random_idx.append(idx)

    random_selection = torch.stack(random_selection)
    u_shuffled = u_patch[batch_range, random_selection]
    labels = torch.tensor(random_idx, device = u.device).squeeze()
    u_out = unpatchify(u_shuffled, patch_size, num_patches, mode)

    return u_out, labels

def shuffle_jigsaw(u, temporal_patch_size, spatial_patch_size, num_classes=64):
    """
    Shuffle samples in a sequence.
    Args:
        u (torch.Tensor): input tensor of shape [batch, time_history, x, (y)]
        temporal_patch_size (int): size of the temporal patch
        spatial_patch_size (int): size of the spatial patch
        num_classes (int, optional): number of classes. Defaults to 50.
    Returns:
        torch.Tensor: shuffled input tensor of shape [batch, time_history, x, (y)]
        torch.Tensor: labels of shape [batch]
    """
    batch_size, time_history, x = u.shape[:3]
    assert time_history % temporal_patch_size == 0, 'time_history must be divisible by temporal_patch_size'
    assert x % spatial_patch_size == 0, 'x must be divisible by spatial_patch_size'

    u_patch, num_patches = patchify(u, temporal_patch_size, mode="jigsaw", jigsaw_patch_size=spatial_patch_size)

    assert num_patches <= 9, f'Number of patches: {num_patches} is too large for full jigsaw, scales with tuple_size factorial'

    all_permutations = get_permutations_hamming(num_patches, num_classes).to(u.device)
    batch_range = torch.arange(batch_size, device = u.device)[:, None]

    # Get (batch_size) number of permutations
    random_selection = []
    random_idx = []
    for idx in np.random.randint(0, len(all_permutations), batch_size):
        random_selection.append(all_permutations[idx])
        random_idx.append(idx)

    random_selection = torch.stack(random_selection)
    u_shuffled = u_patch[batch_range, random_selection]
    labels = torch.tensor(random_idx, device = u.device)
    u_out = unpatchify(u_shuffled, temporal_patch_size, num_patches, mode="jigsaw", jigsaw_img_size=time_history)

    return u_out, labels

def shuffle(u):
    idx = torch.randperm(u.shape[0])
    while torch.all(idx == torch.arange(u.shape[0])):
        idx = torch.randperm(u.shape[0]) # Ensure that the shuffled tensor is not the same as the original
    u = u[idx].view(u.size())
    return u

@lru_cache(maxsize=128)
def get_permutations(seq_len):
    all_permutations = list(itertools.permutations(range(seq_len)))
    return torch.tensor(all_permutations)

@lru_cache(maxsize=128)
def get_permutations_hamming(seq_len, num_classes):
    if num_classes >= math.factorial(seq_len):
        return get_permutations(seq_len)

    all_permutations = list(itertools.permutations(range(seq_len)))

    hamming_dists = torch.zeros(len(all_permutations))
    original_seq = list(range(seq_len))
    for i, seq in enumerate(all_permutations):
        hamming_dists[i] = scipy.spatial.distance.hamming(original_seq, seq)

    idx = np.argsort(-1* hamming_dists.numpy()) # get largest hamming distances 
    out = torch.tensor(all_permutations)[idx[:num_classes]]
    return out



