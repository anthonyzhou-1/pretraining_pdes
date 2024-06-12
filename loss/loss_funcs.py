import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
from torch import distributed as dist

def _diffusion(u, dt, dx, coeff):
    '''
        Calculated 2nd order central difference scheme for diffusion operator
    '''
    diff_u = (torch.roll(u, shifts=(1), dims=(1)) + torch.roll(u, shifts=(-1), dims=(1)) - 2*u)
    #diff_u = diff_u.unsqueeze(-1)
    diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1)**2 
    #print(dx**2/2, dt)
    #raise
    return -diff_u
    

def _advection(u, dt, dx, coeff):
    '''
        Calculated 2nd order central difference scheme for nonlinear advection operator
    '''
    #print("U SHAPE: {}".format(u.shape))
    diff_u = (torch.roll(u, shifts=(1), dims=(1)) - torch.roll(u, shifts=(-1), dims=(1))) * u
    #diff_u = diff_u.unsqueeze(-1)
    #print(coeff.shape, dt.shape, diff_u.shape, dx.shape)
    #print("\nIN ADVECTION\n")
    #print(diff_u.shape)
    #print(coeff.unsqueeze(-1).unsqueeze(-1).shape)
    #print(dt.unsqueeze(1).unsqueeze(1).shape)
    #print(dx.unsqueeze(1).unsqueeze(1).shape)
    diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/(2*dx.unsqueeze(1).unsqueeze(1))
    #print(diff_u.shape)
    #print()
    #diff_u = u*(u - torch.roll(u, shifts=(-1), dims=(1)))
    #diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/(dx.unsqueeze(1).unsqueeze(1))
    return diff_u
    
    
def _dispersion(u, dt, dx, coeff):
    '''
        Calculated 2nd order central difference scheme for third derivative operator
    '''
    numerator =  torch.roll(u, shifts=(3), dims=(1)) - \
               8*torch.roll(u, shifts=(2), dims=(1)) + \
              13*torch.roll(u, shifts=(1), dims=(1)) - \
              13*torch.roll(u, shifts=(-1), dims=(1)) + \
               8*torch.roll(u, shifts=(-2), dims=(1)) - \
                 torch.roll(u, shifts=(-2), dims=(1))
    diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) 
    return diff_u


def _linear_advection(u, dt, dx, coeff):
    '''
        Calculated 2nd order central difference scheme for third derivative operator
    '''
    numerator = u - torch.roll(u, shifts=(-1), dims=(1))
    adv_u = -coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/ \
            dx.unsqueeze(1).unsqueeze(1)
    return adv_u


def _anchored_l2(xi, xj, ui, uj, dt, dx, coeff):
    # Temporal evolution
    future_window = xi.shape[-1]

    # Temporal evolution
    us = (ui[...,-future_window:].unsqueeze(0) - uj[...,-future_window-1:-1].unsqueeze(1))

    # Calculate physics-informed update
    adv = _advection(xi, dt, dx, coeff[:,0])
    diff = _diffusion(xi, dt, dx, coeff[:,1])
    dis = _linear_advection(xi, dt, dx, coeff[:,2])

    updates = (adv + diff + dis)#[...,0]

    # Conservation of mass
    u_mass = ui[...,0].sum(dim=1)
    x_mass = xi.sum(dim=1)[...,0]

    # Distance between different analytical updates to latent space
    physics_distance = ((xi + updates).unsqueeze(1) - xj.unsqueeze(0))

    if(((us-physics_distance).norm(dim=-2)**2).isnan().any()):
        raise ValueError("\n\n\nNAN IN DISTANCE FUNCTION\n\n\n")
    return ((us - physics_distance).norm(dim=(-2)))**2


def _diffusion_2d(u, dt, dx, coeff):
    diff_ux = (torch.roll(u, shifts=(1), dims=(1)) + torch.roll(u, shifts=(-1), dims=(1)) - 2*u)
    diff_uy = (torch.roll(u, shifts=(1), dims=(2)) + torch.roll(u, shifts=(-1), dims=(2)) - 2*u)
    #top = coeff.view(coeff.shape[0], 1, 1, 1)*(diff_ux + diff_uy)
    diff_u = coeff.view(coeff.shape[0], 1, 1, 1)*(diff_ux + diff_uy)/(dx.view(1, 1, 1, 1)**2)
    return diff_u


def _linear_advection_2d(u, dt, dx, coeff):
    '''
        Calculated 2nd order central difference scheme for third derivative operator
    '''
    cx = coeff[:,0].view(coeff.shape[0], 1, 1, 1)
    cy = coeff[:,1].view(coeff.shape[0], 1, 1, 1)
    numerator = u - cx*torch.roll(u, shifts=(-1), dims=(1)) - cy*torch.roll(u, shifts=(-1), dims=(2))
    adv_u = -dt.view(dt.shape[0], 1, 1, 1)*numerator/dx.view(1, 1, 1, 1)
    return adv_u


# TODO Double check that these actually do what they should
def _advection_2d(u, dt, dx, coeff):
    # Calculate finite differences
    un = u.clone()
    vn = u.clone()

    # Calculate finite differences for nonlinear advection term
    # TODO: Vectorize this if possible
    adv_u = torch.zeros(u.shape).to(device=u.device)
    for idx, (cx, cy) in enumerate(coeff):
        cx = cx.view(1,1,1,1)
        cy = cy.view(1,1,1,1)
        if(cx <= 0 and cy >= 0):
            adv_u[idx] = -cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(1))) + \
                          cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(2)))
            #adv_v = cy*vn*(vn - torch.roll(vn, shifts=(1), dims=(0))) - cx*un*(vn - torch.roll(vn, shifts=(-1), dims=(1)))
        elif(cx >= 0 and cy >= 0):
            adv_u[idx] = cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(1))) + \
                         cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(2)))
            #adv_v = cy*vn*(vn - torch.roll(vn, shifts=(1), dims=(0))) + cx*un*(vn - torch.roll(vn, shifts=(1), dims=(1)))
        elif(cx <= 0 and cy <= 0):
            adv_u[idx] = -cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(1))) - \
                          cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(2)))
            #adv_v = -cy*vn*(vn - torch.roll(vn, shifts=(-1), dims=(0))) - cx*un*(vn - torch.roll(vn, shifts=(-1), dims=(1)))
        elif(cx >= 0 and cy <= 0):
            adv_u[idx] = cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(1))) - \
                         cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(2)))
            #adv_v = -cy*vn*(vn - torch.roll(vn, shifts=(-1), dims=(0))) + cx*un*(vn - torch.roll(vn, shifts=(1), dims=(1)))

    # Make shapes broadcastable
    dt = dt.reshape(dt.shape[0], 1, 1, 1)
    dx = dx.view(1, 1, 1, 1)
    return -dt*adv_u/dx


def _anchored_l2_2d(xi, xj, ui, uj, dt, dx, coeff):

    # Temporal evolution
    future_window = xi.shape[-1]

    # Temporal evolution
    #print(ui.shape)
    us = (ui[...,-future_window:].unsqueeze(0) - uj[...,-future_window-1:-1].unsqueeze(1))
    #print(us.shape)

    # Calculate physics-informed update
    #print()
    #print(coeff)
    #print(coeff.shape)
    #print()
    #print(coeff[:,[3,4]])
    #print(coeff[:,[3,4]].shape)
    #print()
    #print(coeff[:,[1,2]])
    #print(coeff[:,[1,2]].shape)
    #print()
    #print(coeff[:,0])
    #print(coeff[:,0].shape)
    #print()
    #raise
    adv = _advection_2d(xi, dt, dx, coeff[:,[3, 4]])
    diff = _diffusion_2d(xi, dt, dx, coeff[:,0])
    linadv = _linear_advection_2d(xi, dt, dx, coeff[:,[1, 2]])
    #print(adv.shape)
    #print(diff.shape)
    #print(dis.shape)
    #raise

    updates = (adv + diff + linadv)#[...,0]

    # Conservation of mass
    u_mass = ui[...,0].sum(dim=1)
    x_mass = xi.sum(dim=1)[...,0]

    # Distance between different analytical updates to latent space
    physics_distance = ((xi + updates).unsqueeze(1) - xj.unsqueeze(0))

    #print(us.shape, physics_distance.shape)
    #print(us.isnan().any())
    #print(adv.isnan().any(), diff.isnan().any(), linadv.isnan().any())
    #print(physics_distance.isnan().any())
    #print((us-physics_distance).norm(dim=(2,3)).shape)
    #raise

    if(((us-physics_distance).norm(dim=(2,3))**2).isnan().any()):
        raise ValueError("\n\n\nNAN IN DISTANCE FUNCTION\n\n\n")
    return ((us - physics_distance).norm(dim=(2,3)))**2



#def _anchored_l2_2d(xi, xj, ui, uj, dt, dx, coeff):
#
#    # Temporal evolution
#    us = (ui[...,-1].unsqueeze(0) - uj[...,-2].unsqueeze(1)).norm(dim=(-2,-1)) # Temporal evolution
#
#    difference = ((xi[...,-1] - ui[...,-1]).unsqueeze(0) - \
#                  (xj[...,-1] - uj[...,-1]).unsqueeze(1)).norm(dim=-1)
#
#    # Calculate physics-informed update
#    #print(coeff)
#    diff = _diffusion_2d(xi, dt, dx, coeff[:,0])
#    adv = _advection_2d(xi, dt, dx, coeff[:,1:])
#    #advy = _advection(xi, dt, dx, coeff[:,0])
#    #dis = _dispersion(xi, dt, dx, coeff[:,2])
#
#    if(diff.isnan().any()):
#        print("ISSUE WITH DIFF")
#    if(adv.isnan().any()):
#        print("ISSUE WITH ADV")
#    updates = (adv + diff)[...,0]
#
#    # Conservation of mass
#    u_mass = ui[...,0].sum(dim=1)
#    x_mass = xi.sum(dim=1)[...,0]
#
#    # Homogeneous case
#    #mass = (u_mass.unsqueeze(0) - x_mass.unsqueeze(1))*dx
#
#    # Operator distance TODO: This is not what I want it to be...
#    #physics_distance = (updates.unsqueeze(1) - updates.unsqueeze(0)).norm(dim=(-2,-1))
#    #print(xi.shape)
#    #print(updates.shape)
#    physics_distance = ((xi[...,0] + updates).unsqueeze(1) - xj[...,0].unsqueeze(0)).norm(dim=(-2,-1))
#    #print()
#    #print()
#    #print(updates.isnan().any())
#    #print(physics_distance)
#    #print(us.isnan().any())
#    #print(us)
#    #print(physics_distance.shape)
#    #print()
#    #print()
#
#    #return (us - physics_distance)**2 + mass**2 + difference**2  # Full
#    #return (us - physics_distance)**2 + difference**2            # No mass
#    #return (us - physics_distance)**2 + mass**2                  # No pred
#    #return mass**2 + difference**2                                # No Physics
#    #print((0.1*us - physics_distance)**2)
#    return (us - physics_distance)**2                            # Just Physics
#    #return mass**2                                                # Just mass
#    #return difference**2                                          # Just pred


class WeightedNTXentLoss(torch.nn.Module):
    def __init__(self, device, temperature=1., similarity='cosine', lambda_1=0.9, **kwargs):
        super(WeightedNTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        #self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.similarity_function = self._get_similarity_function(similarity)
        self.lambda_1 = lambda_1
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, similarity):
        #return torch.cdist
        #f = lambda xi, xj, ui, uj: (frdist(xi, xj) - frdist(ui, uj)).abs()
        if(similarity == 'cosine'):
            print("\nUSING COSINE SIMILARITY\n")
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        elif(similarity == 'anchored_l2'):
            print("\nUSING ANCHORED L2 SIMILARITY\n")
            return _anchored_l2
        elif(similarity == 'dot'):
            print("\nUSING DOT SIMILARITY\n")
            return self._dot_simililarity
        else:
            raise ValueError("Invalid choice of similarity function.")
        #if use_cosine_similarity:
        #    self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        #    return self._cosine_simililarity
        #else:
        #    return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        #diag = np.eye(batch_size)
        #l1 = np.eye((batch_size), batch_size, k=-batch_size)
        #l2 = np.eye((batch_size), batch_size, k=batch_size)
        #mask = torch.from_numpy((diag + l1 + l2))
        #mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        print(v.shape)
        raise
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        #print(x.shape)
        #print(y.shape)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        #print(v.shape)
        return v


    def _anchored_l2(self, xi, xj, ui, uj, dt, dx, coeff):
        xs = (xi[...,0].unsqueeze(0) - xj[...,0].unsqueeze(1)).norm(dim=-1)#[...,0]
        us = (ui.unsqueeze(0) - uj.unsqueeze(1)).norm(dim=-1)#[...,0]

        adv = self._advection(xi, dt, dx, coeff[:,0])
        diff = self._diffusion(xi, dt, dx, coeff[:,1])
        dis = self._dispersion(xi, dt, dx, coeff[:,2])

        updates = (adv + diff + dis)[...,0]
        operator_distance = (updates.unsqueeze(1) - updates.unsqueeze(0)).norm(dim=-1)
        #print((xs - us).abs().shape, operator_distance.shape)
        #raise
        return -((xs - us).abs() + operator_distance)


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        diff_u = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1)**2 
        return diff_u
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        diff_u = (-torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1))) * u
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1) 
        return diff_u
        
        
    def _dispersion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for third derivative operator
        '''
        numerator =  torch.roll(u, shifts=(0,3), dims=(0,1)) - \
                   8*torch.roll(u, shifts=(0,2), dims=(0,1)) + \
                  13*torch.roll(u, shifts=(0,1), dims=(0,1)) - \
                  13*torch.roll(u, shifts=(0,-1), dims=(0,1)) + \
                   8*torch.roll(u, shifts=(0,-2), dims=(0,1)) - \
                     torch.roll(u, shifts=(0,-2), dims=(0,1))
        diff_u = coeff.unsqueeze(-1).unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) 
        return diff_u


    def _pde_similarity(self, u, dt, dx, target):
        
        u_adv = self._advection(u, dt, dx, target[:,0].unsqueeze(-1))
        u_diff = self._diffusion(u, dt, dx, target[:,1].unsqueeze(-1))
        u_third = self._third_order(u, dt, dx, target[:,2].unsqueeze(-1))

        xi = torch.cat((u_adv, u_diff, u_third), dim=1)
        xj = torch.cat((u_adv, u_diff, u_third), dim=1)

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return prod_mat / norm_mat


    def _pde_similarity(self, target):
        xi = target.clone()
        xj = target.clone()

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]
        #print(prod_mat.shape)
        #print(norm_mat.shape)
        #print(prod_mat[0][0], norm_mat[0][0])
        #print(prod_mat)
        #print(norm_mat)
        #raise
        return prod_mat / norm_mat


    def forward(self, x1, x2, u, dt, dx, vs):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim = self._pde_similarity(vs)
        sim_score = 1 - self.lambda_1 * sim
        sim_score = torch.nan_to_num(sim_score, nan=0.)
        sim_score = sim_score.masked_select(~torch.eye(batch_size, dtype=bool)).view(batch_size, batch_size - 1)
        sim_score = sim_score.repeat(2,2)

        representations = torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        #similarity_matrix = self.similarity_function(representations[...,0],
                                                     #representations[...,0])
        #similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
                                                     #representations.flatten(1,2))#[...,0])

        #similarity_matrix = self.similarity_function(x1.flatten(1,2), x2.flatten(1,2))
        #print()
        #print()
        #print(x1.shape, u[0].shape)
        #print()
        #print()
        if(len(x1.shape) == 4):
            x1 = x1[...,0]
        if(len(x2.shape) == 4):
            x2 = x2[...,0]

        u = torch.cat((u,u), dim=0)
        dt = torch.cat((dt,dt), dim=0)
        vs = torch.cat((vs,vs), dim=0)
        similarity_matrix = self.similarity_function(representations.flatten(1,2), representations.flatten(1,2),
                                                     u.clone(), u.clone(), dt, dx, vs)
                                                     #u[...,0].clone(), u[...,0].clone(), dt, dx, vs)

        #print(similarity_matrix.shape)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        #print(positives)
        
        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #print(mask_samples_from_same_repr.shape)
        #print(similarity_matrix.shape)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        #print(negatives.shape)
        #print(sim_score.shape)
        negatives *= sim_score
        
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        
        return loss / (2 * batch_size)
        #similarity_matrix = torch.zeros(batch_size, batch_size)
        #for i in range(len(x1)):
        #    for j in range(i, len(x2)):
        #        sim = self.similarity_function(x1[i], x1[j], u[i], u[j])
        #        similarity_matrix[i,j] = sim
        #        similarity_matrix[j,i] = sim

        #similarity_matrix = self.similarity_function(x1.flatten(1,2), x2.flatten(1,2),
        #                                             u.flatten(1,2).clone(), u.flatten(1,2).clone())
        #similarity_matrix = torch.Tensor(similarity_matrix)
        #print(similarity_matrix)
        #print(representations.shape)
        #print(x1.shape)
        #print(x2.shape)
        #print(similarity_matrix.shape)
        #print(l_pos)
        #print(similarity_matrix.shape)
        #raise
        #l_pos = torch.diag(similarity_matrix, batch_size)
        #r_pos = torch.diag(similarity_matrix, -batch_size)
        #l_pos = torch.diag(similarity_matrix)#, batch_size)
        #r_pos = torch.diag(similarity_matrix)#, -batch_size)
        #print(l_pos)
        #print(r_pos)
        #print(l_pos.shape)
        #print(r_pos.shape)
        #positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        #positives = torch.cat([l_pos, r_pos]).unsqueeze(-1)#.view(batch_size, 1)
        #positives = l_pos.clone().unsqueeze(-1)
        #print(x1)
        #print(x2)
        #print(similarity_matrix)
        #raise

        #mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        #mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #negatives = similarity_matrix[mask_samples_from_same_repr].view(batch_size, -1)
        #print(negatives.shape)
        #print(sim_score)
        #print(negatives)
        #print(positives)
        #raise
        #negatives *= sim_score[mask_samples_from_same_repr].view(2*batch_size, -1)
        #print(sim_score.shape)
        #print(sim_score)
        #negatives *= sim_score[mask_samples_from_same_repr].view(batch_size, -1)
        #raise

        #print(positives.shape)
        #print(negatives.shape)
        #logits = torch.cat((positives, negatives), dim=1)
        #logits /= self.temperature
        #print(logits)

        #labels = torch.zeros(2 * batch_size).to(self.device).long()
        #labels = torch.zeros(batch_size).to(self.device).long()
        #print(labels)
        #print(logits)
        #loss = self.criterion(logits, labels)
        #print(logits)
        #print(loss)
        #return loss/batch_size
        #raise

        # filter out the scores from the positive samples
        #print(similarity_matrix)

        # filter out the scores from the positive samples
        #l_pos = torch.diag(similarity_matrix, batch_size)
        #r_pos = torch.diag(similarity_matrix, -batch_size)
        #positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        #mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        #negatives *= sim_score

        #logits = torch.cat((positives, negatives), dim=1)
        #logits /= self.temperature

        #labels = torch.zeros(2 * batch_size).to(self.device).long()
        #loss = self.criterion(logits, labels)

        #return loss / (2 * batch_size)


class PhysicsInformedWNTXentLoss(torch.nn.Module):
    def __init__(self, device, temperature=0.1, use_cosine_similarity=True, lambda_1=0.9999, **kwargs):
        super(PhysicsInformedWNTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.lambda_1 = lambda_1
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, use_cosine_similarity):
        print("\nUSING DOT SIMILARITY\n")
        return self._dot_simililarity
        #return torch.cdist
        #if use_cosine_similarity:
        #    self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        #    return self._cosine_simililarity
        #else:
        #    return self._dot_simililarity


    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        diff_u = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1)**2 + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        diff_u = (-torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1))) * u
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1) + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        
        
    def _third_order(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for third derivative operator
        '''
        #numerator = torch.roll(u, shifts=(0,2), dims=(0,1)) - \
        #           2*torch.roll(u, shifts=(0,1), dims=(0,1)) + \
        #           2*torch.roll(u, shifts=(0,-1), dims=(0,1)) - \
        #             torch.roll(u, shifts=(0,-2), dims=(0,1))
        numerator =  torch.roll(u, shifts=(0,3), dims=(0,1)) - \
                   8*torch.roll(u, shifts=(0,2), dims=(0,1)) + \
                  13*torch.roll(u, shifts=(0,1), dims=(0,1)) - \
                  13*torch.roll(u, shifts=(0,-1), dims=(0,1)) + \
                   8*torch.roll(u, shifts=(0,-2), dims=(0,1)) - \
                     torch.roll(u, shifts=(0,-2), dims=(0,1))
        #
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)


    def _pde_similarity(self, u, dt, dx, target):
        
        u_adv = self._advection(u, dt, dx, target[:,0].unsqueeze(-1))
        u_diff = self._diffusion(u, dt, dx, target[:,1].unsqueeze(-1))
        u_third = self._third_order(u, dt, dx, target[:,2].unsqueeze(-1))
        #print(target[:,2])
        #print()
        #print(u_third)
        #print()

        xi = torch.cat((u_adv, u_diff, u_third), dim=1)
        xj = torch.cat((u_adv, u_diff, u_third), dim=1)
        #print(xi)
        #print(target)
        #raise

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        #print()
        #print(prod_mat)
        #print(norm_mat)
        #print(target)
        #print(xi)
        #print(prod_mat/norm_mat)
        #print()
        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        #diag = np.eye(2 * batch_size)
        #l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        #l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        #mask = torch.from_numpy((diag + l1 + l2))
        #mask = (1 - mask).type(torch.bool)
        diag = np.eye(batch_size)
        l1 = np.eye((batch_size), batch_size, k=-batch_size)
        l2 = np.eye((batch_size), batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        #sim_score = np.zeros((batch_size, batch_size-1))
        #for i in range(len(vs)):
        #    for j in range(i+1, len(vs)):
        #        sim = self._similarity(vs[i], vs[j])
        #        sim_score[i,j-1] = sim
        #        sim_score[j,i] = sim

        #sim_score = 1 - self.lambda_1 * torch.tensor(sim_score, dtype=torch.float).to(x1.device)
        #sim_score = sim_score.repeat(2, 2)
        #sim_score = torch.tensor(sim_score, dtype=torch.float).repeat(2, 2)


        #print(vs[:4])
        #sim = self._pde_similarity(vs)
        sim = self._pde_similarity(u, dt, dx, vs)
        #print(sim)
        #print(sim.max())
        #print(sim.min())
        #raise
        #print(sim)
        sim_score = 1 - self.lambda_1 * sim
        #sim_score = sim.repeat(2, 2)
        sim_score = torch.nan_to_num(sim_score, nan=0.)
        #print(sim_score)
        #raise


        representations = torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        #similarity_matrix = self.similarity_function(representations[...,0],
                                                     #representations[...,0])
        #similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
        #                                             representations.flatten(1,2))#[...,0])
        similarity_matrix = self.similarity_function(x1.flatten(1,2), x2.flatten(1,2))
        #print(representations.shape)
        #print(x1.shape)
        #print(x2.shape)
        #print(similarity_matrix.shape)
        #print(l_pos)
        #print(similarity_matrix.shape)
        #raise
        #l_pos = torch.diag(similarity_matrix, batch_size)
        #r_pos = torch.diag(similarity_matrix, -batch_size)
        l_pos = torch.diag(similarity_matrix)#, batch_size)
        #r_pos = torch.diag(similarity_matrix)#, -batch_size)
        #print(l_pos)
        #print(r_pos)
        #print(l_pos.shape)
        #print(r_pos.shape)
        #positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        #positives = torch.cat([l_pos, r_pos]).unsqueeze(-1)#.view(batch_size, 1)
        positives = l_pos.clone().unsqueeze(-1)
        #print(x1)
        #print(x2)
        #print(similarity_matrix)
        #raise

        #mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(batch_size, -1)
        #print(negatives.shape)
        #print(sim_score)
        #print(negatives)
        #raise
        #negatives *= sim_score[mask_samples_from_same_repr].view(2*batch_size, -1)
        #print(sim_score.shape)
        #print(sim_score)
        negatives *= sim_score[mask_samples_from_same_repr].view(batch_size, -1)
        #raise

        #print(positives.shape)
        #print(negatives.shape)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        #labels = torch.zeros(2 * batch_size).to(self.device).long()
        labels = torch.zeros(batch_size).to(self.device).long()
        #print(labels)
        #print(logits)
        loss = self.criterion(logits, labels)
        #print(logits)
        #print(loss)
        return loss/batch_size
        #raise

        # filter out the scores from the positive samples
        #print(similarity_matrix)

        # filter out the scores from the positive samples
        #l_pos = torch.diag(similarity_matrix, batch_size)
        #r_pos = torch.diag(similarity_matrix, -batch_size)
        #positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        #mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        #negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        #negatives *= sim_score

        #logits = torch.cat((positives, negatives), dim=1)
        #logits /= self.temperature

        #labels = torch.zeros(2 * batch_size).to(self.device).long()
        #loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


class PhysicsInformedGCL(torch.nn.Module):
    def __init__(self, device, tau=0.5, use_cosine_similarity=True, **kwargs):
        super(PhysicsInformedGCL, self).__init__()
        #self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.tau = tau
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, use_cosine_similarity):
        print("\nUSING COSINE SIMILARITY\n")
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity


    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        diff_u = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1)**2 + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(0.01*(u[...,1] - diff_u[...,0]).abs(), dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        diff_u = (-torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1))) * u
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1) + u
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print()
        #print(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1))
        #print(torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0])
        #print(coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - coeff*diff_u[...,0], dim=1)**2)))
        #print(coeff[:,0]*torch.exp(-(torch.sum(0.01*(u[...,1] - coeff*diff_u[...,0]).abs(), dim=1)**2)))
        #print(torch.exp(-(torch.sum(0.01*(u[...,1] - coeff*diff_u[...,0]).abs(), dim=1)**2)))
        #raise
        #print()
        return (coeff[:,0]*torch.exp(-(torch.sum(0.01*(u[...,1] - diff_u[...,0]).abs(), dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        
        
    def _third_order(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for third derivative operator
        '''
        #numerator = torch.roll(u, shifts=(0,2), dims=(0,1)) - \
        #           2*torch.roll(u, shifts=(0,1), dims=(0,1)) + \
        #           2*torch.roll(u, shifts=(0,-1), dims=(0,1)) - \
        #             torch.roll(u, shifts=(0,-2), dims=(0,1))
        numerator =  torch.roll(u, shifts=(0,3), dims=(0,1)) - \
                   8*torch.roll(u, shifts=(0,2), dims=(0,1)) + \
                  13*torch.roll(u, shifts=(0,1), dims=(0,1)) - \
                  13*torch.roll(u, shifts=(0,-1), dims=(0,1)) + \
                   8*torch.roll(u, shifts=(0,-2), dims=(0,1)) - \
                     torch.roll(u, shifts=(0,-2), dims=(0,1))
        #
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) + u
        #print(u[...,1])
        #print(diff_u[...,0])
        #print(torch.sum((u[...,1] - diff_u[...,0]).abs(), dim=1))
        return (coeff[:,0]*torch.exp(-(torch.sum(0.01*(u[...,1] - diff_u[...,0]).abs(), dim=1)**2))).unsqueeze(1)
        #return (coeff*torch.sum(u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)
        #return (coeff*torch.sum(diff_u[...,1] - diff_u[...,0], dim=1)).unsqueeze(1)


    def _pde_similarity(self, u, dt, dx, target):
        
        u_adv = self._advection(u, dt, dx, target[:,0].unsqueeze(-1))
        u_diff = self._diffusion(u, dt, dx, target[:,1].unsqueeze(-1))
        u_third = self._third_order(u, dt, dx, target[:,2].unsqueeze(-1))
        #print(target[:,2])
        #print()
        #print(u_third)
        #print()

        xi = torch.cat((u_adv, u_diff, u_third), dim=1)
        xj = torch.cat((u_adv, u_diff, u_third), dim=1)
        #print(target[:,-1])
        #print(u_third)
        #raise
        #print(target)
        #print(xi)
        #print(xi)
        #print(target)

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        #print()
        #print(prod_mat)
        #print(norm_mat)
        #print(target)
        #print(xi)
        #print(prod_mat/norm_mat)
        #print()
        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim_score = np.zeros((batch_size, batch_size-1))

        sim_score = self._pde_similarity(u, dt, dx, vs)

        representations = x1#torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        #print(representations.shape)
        #similarity_matrix = self.similarity_function(representations[...,0],
        #                                             representations[...,0])
        similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
                                                     representations.flatten(1,2))#[...,0])
        #print(similarity_matrix)
        #print(similarity_matrix.shape)

        positive = 0.5 * sim_score * similarity_matrix**2
        #torch.cat(((self.tau-similarity_matrix**2).unsqueeze(0), torch.zeros(sim_score.shape).unsqueeze(0))).shape)
        negative = 0.5 * (1 - sim_score) * torch.max(torch.cat(((self.tau - similarity_matrix**2).unsqueeze(0),
                                                           torch.zeros(sim_score.shape).unsqueeze(0)), axis=0))
        return (positive + negative).mean()
        raise

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        negatives *= sim_score[mask_samples_from_same_repr].view(2*batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        if(loss.isnan()):
            print(similarity_matrix)
            print(sim_score)
            print(logits)

        return loss #/ (2 * batch_size)


class GCL(torch.nn.Module):
    def __init__(self, device, tau=10., similarity_fn='anchored_l2', **kwargs):
        super(GCL, self).__init__()
        self.device = device
        self.similarity_function = self._get_similarity_function(similarity_fn)
        self.tau = tau
        self._tau = tau
        print("\nTAU: {}".format(self.tau))
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, similarity_fn):
        #print("\nUSING ANCHORED L2 SIMILARITY\n")
        if(similarity_fn == 'anchored_l2'):
            return _anchored_l2
        else:
            raise NotImplementedError("Only Anchored L2 Similarity is supported.")


    def _pde_similarity(self, u, dt, dx, target):
        xi = target.clone()
        xj = target.clone()

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0], device=u.device).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0], device=u.device).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return prod_mat / norm_mat


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim_score = np.zeros((batch_size, batch_size-1))
        sim_score = self._pde_similarity(u, dt, dx, vs).cuda()
        sim_score = sim_score.repeat(2,2)

        #representations = x1#torch.cat([x2, x1], dim=0)
        representations = torch.cat([x2, x1], dim=0)

        if(len(x1.shape) == 4):
            x1 = x1[...,0]
        if(len(x2.shape) == 4):
            x2 = x2[...,0]

        u = torch.cat((u,u), dim=0)
        dt = torch.cat((dt,dt), dim=0)
        vs = torch.cat((vs,vs), dim=0)
        #print()
        #print()
        #print(representations.shape)
        #print(representations.device)
        #print(representations.flatten(1,2).shape)
        #print()
        #print()
        #similarity_matrix = self.similarity_function(representations.flatten(1,2).cuda(), representations.flatten(1,2).cuda(),
        similarity_matrix = self.similarity_function(representations, representations,
                                                     u.clone().cuda(), u.clone().cuda(),
                                                     dt.cuda(), dx.cuda(), vs.cuda()).cuda()
        #print(similarity_matrix.shape)
        #print(sim_score.unsqueeze(-1).broadcast_to(similarity_matrix.shape).shape)
        sim_score = sim_score.unsqueeze(-1).broadcast_to(similarity_matrix.shape)
        #raise
        
        positive = 0.5 * sim_score * similarity_matrix**2

        if(self._tau == 'mean'):
            self.tau = similarity_matrix.mean().cuda()
        negative = 0.5 * (1 - sim_score) * torch.max(torch.cat(((self.tau - similarity_matrix**2).unsqueeze(0),
                                                           torch.zeros(sim_score.shape).unsqueeze(0).cuda()), axis=0))
        return (positive + negative).mean()/batch_size


class GCL2D(torch.nn.Module):
    def __init__(self, device, tau=100., similarity_fn='anchored_l2_2d', **kwargs):
        super(GCL2D, self).__init__()
        self.device = device
        self.similarity_function = self._get_similarity_function(similarity_fn)
        self.tau = tau
        self._tau = tau
        print("\nTAU: {}".format(self.tau))
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, similarity_fn):
        if(similarity_fn == 'anchored_l2_2d'):
            return _anchored_l2_2d
        else:
            raise NotImplementedError("Only Anchored L2 Similarity is supported.")


    def _pde_similarity_2d(self, u, dt, dx, target):
        
        # Convert individual advections into total advection
        true_target = torch.zeros((len(target), 2)).to(device=u.device)
        true_target[:,0] = target[:,0]
        true_target[:,1] = target[:,1:].norm(dim=1)
        xi = true_target.clone()
        xj = true_target.clone()

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).to(device=u.device).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).to(device=u.device).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim_score = self._pde_similarity_2d(u, dt, dx, vs)
        sim_score = sim_score.repeat(2,2)

        representations = torch.cat([x2, x1], dim=0)

        # Normalizing representations seems to help with stability
        #representations = (representations - representations.mean())/representations.std()

        ###
        # TODO: Double check this is doing what I want
        ###
        #print(representations.shape)
        #similarity_matrix = self.similarity_function(representations[...,0],
        #                                             representations[...,0])
        #similarity_matrix = self.similarity_function(representations.flatten(1,2),#[...,0],
        #                                             representations.flatten(1,2))#[...,0])
        if(len(x1.shape) == 4):
            x1 = x1[...,0]
        if(len(x2.shape) == 4):
            x2 = x2[...,0]

        u = torch.cat((u,u), dim=0)
        dt = torch.cat((dt,dt), dim=0)
        vs = torch.cat((vs,vs), dim=0)
        similarity_matrix = self.similarity_function(representations, representations,
                                                     u.clone(), u.clone(), dt, dx, vs)
        
        # Normalize similarity matrix
        #similarity_matrix = (similarity_matrix - similarity_matrix.mean())/similarity_matrix.std()

        sim_score = sim_score.unsqueeze(-1).broadcast_to(similarity_matrix.shape)

        positive = 0.5 * sim_score * similarity_matrix**2

        if(self._tau == 'mean'):
            self.tau = similarity_matrix.mean().cuda()
        negative = 0.5 * (1 - sim_score) * torch.max(torch.cat(((self.tau - similarity_matrix**2).unsqueeze(0),
                                                           torch.zeros(sim_score.shape).unsqueeze(0).cuda()), axis=0))
        return (positive + negative).mean()/batch_size


class VICReg(nn.Module):
    def __init__(self, sim_coef=1., std_coef=1., cov_coef=1.):
        super().__init__()
        #self.args = args
        self.sim_coef = sim_coef
        self.std_coef = std_coef
        self.cov_coef = cov_coef
        #self.num_features = int(args.mlp.split("-")[-1])
        #self.backbone, self.embedding = resnet.__dict__[args.arch](
        #    zero_init_residual=True
        #)
        #self.projector = Projector(args, self.embedding)


    def _pde_similarity(self, target):
        
        xi = target.clone()
        xj = target.clone()

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return prod_mat / norm_mat

    
    def forward(self, x, y, target):
        #TODO: Batch this
        #x = self.projector(self.backbone(x))
        #y = self.projector(self.backbone(y))
    
        sim_score = self._pde_similarity(target)
        print(sim_score)
        print(sim_score.shape)
        print(x.shape)
        print(y.shape)
        repr_loss = F.mse_loss(x, y)
        repr_loss = torch.mean(sim_score[0] * (x - y)**2)
        print(repr_loss)
    
        #x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #y = torch.cat(FullGatherLayer.apply(y), dim=0)
        print(x.shape)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        raise
    
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    
        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
    
        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


class PhysicsInformedWNTXentLossNS(torch.nn.Module):
    def __init__(self, device, temperature=0.1, use_cosine_similarity=True, lambda_1=0.1, **kwargs):
        super(PhysicsInformedWNTXentLossNS, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.lambda_1 = lambda_1
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity


    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        # Calculate finite differences
        diff_ux = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        diff_uy = (torch.roll(u, shifts=(0, 1), dims=(0,2)) + torch.roll(u, shifts=(0, -1), dims=(0,2)) - 2*u)
        diff_u = diff_ux + diff_uy

        # Make everything the correct shape
        dt = dt.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        dx = dx.unsqueeze(-1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        coeff = coeff.unsqueeze(-1)

        # Calculate update
        diff_u = coeff*dt*diff_u/dx**2 + u

        # Weight
        return (coeff[:,0,0,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=(1,2))**2))).unsqueeze(1)


    def _force(self, u, dt, dx, coeff):
        #coeff.unsqueeze(-1)*
        raise
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        diff_u = (-torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1))) * u
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*diff_u/dx.unsqueeze(1).unsqueeze(1) + u
        return (coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2))).unsqueeze(1)
        
        
    def _third_order(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for third derivative operator
        '''
        numerator =  torch.roll(u, shifts=(0,3), dims=(0,1)) - \
                   8*torch.roll(u, shifts=(0,2), dims=(0,1)) + \
                  13*torch.roll(u, shifts=(0,1), dims=(0,1)) - \
                  13*torch.roll(u, shifts=(0,-1), dims=(0,1)) + \
                   8*torch.roll(u, shifts=(0,-2), dims=(0,1)) - \
                     torch.roll(u, shifts=(0,-2), dims=(0,1))
        #
        diff_u = coeff.unsqueeze(-1)*dt.unsqueeze(1).unsqueeze(1)*numerator/(2*dx.unsqueeze(1).unsqueeze(1)**3) + u
        return (coeff[:,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=1)**2))).unsqueeze(1)


    def _pde_similarity(self, u, dt, dx, target):
        
        #u_adv = self._advection(u, dt, dx, target[:,0].unsqueeze(-1))
        #u_force = self._forcing_term(u, dt, dx, target[:,0].unsqueeze(-1))
        u_diff = self._diffusion(u, dt, dx, target[:,1].unsqueeze(-1))
        #u_third = self._third_order(u, dt, dx, target[:,2].unsqueeze(-1))

        #print("U DIFF SHAPE: {}".format(u_diff.shape))
        #xi = torch.cat((u_adv, u_diff, u_third), dim=1)
        #xj = torch.cat((u_adv, u_diff, u_third), dim=1)
        xi = u_diff.clone()
        xj = u_diff.clone()

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        #print()
        #print(prod_mat)
        #print(norm_mat)
        #print(target)
        #print(xi)
        #print(prod_mat/norm_mat)
        #print()
        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim_score = np.zeros((batch_size, batch_size-1))

        sim = self._pde_similarity(u, dt, dx, vs)
        #print(sim)
        #print(sim_score.shape)
        #print(sim_score)

        #sim_score = 1 - self.lambda_1 * torch.tensor(sim, dtype=torch.float).to(x1.device)
        sim_score = 1 - self.lambda_1 * sim
        sim_score = sim_score.repeat(2, 2)
        sim_score = torch.nan_to_num(sim_score, nan=0.)
        #print(sim_score.shape)
        #print(sim_score)
        #raise

        representations = torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        similarity_matrix = self.similarity_function(representations[...,0],
                                                     representations[...,0])

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        negatives *= sim_score[mask_samples_from_same_repr].view(2*batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        if(loss.isnan()):
            print(similarity_matrix)
            print(sim_score)
            print(logits)

        return loss #/ (2 * batch_size)


class PhysicsInformedWNTXentLoss2D(torch.nn.Module):
    def __init__(self, device, temperature=0.1, use_cosine_similarity=True, lambda_1=0.1, **kwargs):
        super(PhysicsInformedWNTXentLoss2D, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.lambda_1 = lambda_1
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")


    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity


    def _diffusion(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        # Calculate finite differences
        #diff_ux = (torch.roll(u, shifts=(0, 1), dims=(0,1)) + torch.roll(u, shifts=(0, -1), dims=(0,1)) - 2*u)
        #diff_uy = (torch.roll(u, shifts=(0, 1), dims=(0,2)) + torch.roll(u, shifts=(0, -1), dims=(0,2)) - 2*u)
        #diff_u = diff_ux + diff_uy
        un = u.clone()
        diff_ux = (torch.roll(un, shifts=(1), dims=(1)) + torch.roll(un, shifts=(-1), dims=(1)) - 2*un)
        diff_uy = (torch.roll(un, shifts=(1), dims=(0)) + torch.roll(un, shifts=(-1), dims=(0)) - 2*un)
        diff_u = diff_ux + diff_uy

        # Make everything the correct shape
        dt = dt.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        dx = dx.unsqueeze(-1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        coeff = coeff.unsqueeze(-1).unsqueeze(-1)

        # Calculate update
        diff_u = coeff*dt*diff_u/dx**2 + u

        # Weight
        return (coeff[:,0,0,0]*torch.exp(-(torch.sum(u[...,1] - diff_u[...,0], dim=(1,2))**2))).unsqueeze(1)


    def _ns_diffusion(self, vx, vy, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for diffusion operator
        '''
        # Calculate finite differences
        un = vx.clone()
        vn = vy.clone()
        diff_ux = (torch.roll(un, shifts=(1), dims=(1)) + torch.roll(un, shifts=(-1), dims=(1)) - 2*un)
        diff_uy = (torch.roll(vn, shifts=(1), dims=(0)) + torch.roll(vn, shifts=(-1), dims=(0)) - 2*vn)
        diff_u = diff_ux + diff_uy

        # Make everything the correct shape
        dt = dt.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        dx = dx.unsqueeze(-1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        coeff = coeff.unsqueeze(-1).unsqueeze(-1)

        # TODO: Double check this is valid mathematically
        # Calculate update
        diff_vx = coeff*dt*diff_u/dx**2 + vx
        diff_vy = coeff*dt*diff_u/dx**2 + vy

        # Weight
        return 0.5*((coeff[:,0,0,0]*torch.exp(-(torch.sum(vx[...,1] - diff_vx[...,0], dim=(1,2))**2))).unsqueeze(1) + \
                    (coeff[:,0,0,0]*torch.exp(-(torch.sum(vy[...,1] - diff_vy[...,0], dim=(1,2))**2))).unsqueeze(1))
        
    
    def _advection(self, u, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        v = u.clone()
        un = u.clone()
        vn = u.clone()
        adv_u = torch.empty(u.shape)
        for idx in range(u.shape[0]):
            cx = coeff[idx][0]
            cy = coeff[idx][1]
            if(cx <= 0 and cy >= 0):
                adv_u[idx] = -cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(1))) + \
                              cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(0)))
            elif(cx >= 0 and cy >= 0):
                adv_u[idx] = cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(1))) + \
                             cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(0)))
            elif(cx <= 0 and cy <= 0):
                adv_u[idx] = -cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(1))) - \
                              cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(0)))
            elif(cx >= 0 and cy <= 0):
                adv_u[idx] = cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(1))) - \
                             cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(0)))

        # Calculate update
        adv_u = u - dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*adv_u/dx

        return (coeff.norm(dim=1)[:,0]*torch.exp(-(torch.sum(u[...,1] - adv_u[...,0], dim=(1,2))**2))).unsqueeze(1)
        

    def _ns_advection(self, vx, vy, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        un = vx.clone()
        vn = vy.clone()
        adv_u = torch.empty(vx.shape)
        adv_v = torch.empty(vy.shape)
        for idx in range(vx.shape[0]):
            cx = coeff[idx][0]
            cy = coeff[idx][1]
            if(cx <= 0 and cy >= 0):
                adv_u[idx] = -cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(1))) + \
                              cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(0)))
            elif(cx >= 0 and cy >= 0):
                adv_u[idx] = cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(1))) + \
                             cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(0)))
            elif(cx <= 0 and cy <= 0):
                adv_u[idx] = -cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(1))) - \
                              cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(0)))
            elif(cx >= 0 and cy <= 0):
                adv_u[idx] = cx*un[idx]*(un[idx] - torch.roll(un[idx], shifts=(1), dims=(1))) - \
                             cy*vn[idx]*(un[idx] - torch.roll(un[idx], shifts=(-1), dims=(0)))

            if(cy <= 0 and cx >= 0):
                adv_v[idx] = -cy*vn[idx]*(vn[idx] - torch.roll(vn[idx], shifts=(-1), dims=(1))) + \
                              cx*un[idx]*(vn[idx] - torch.roll(vn[idx], shifts=(1), dims=(0)))
            elif(cy >= 0 and cx >= 0):
                adv_v[idx] = cy*un[idx]*(vn[idx] - torch.roll(vn[idx], shifts=(1), dims=(1))) + \
                             cx*vn[idx]*(vn[idx] - torch.roll(vn[idx], shifts=(1), dims=(0)))
            elif(cy <= 0 and cx <= 0):
                adv_v[idx] = -cy*un[idx]*(vn[idx] - torch.roll(vn[idx], shifts=(-1), dims=(1))) - \
                              cx*vn[idx]*(vn[idx] - torch.roll(vn[idx], shifts=(-1), dims=(0)))
            elif(cy >= 0 and cx <= 0):
                adv_v[idx] = cy*un[idx]*(vn[idx] - torch.roll(vn[idx], shifts=(1), dims=(1))) - \
                             cx*vn[idx]*(vn[idx] - torch.roll(vn[idx], shifts=(-1), dims=(0)))

        # Calculate update
        adv_u = vx - dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*adv_u/dx
        adv_v = vy - dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*adv_v/dx

        return torch.cat((coeff[:,0]*torch.exp(-(torch.sum(vx[...,1] - adv_u[...,0], dim=(1,2))**2)).unsqueeze(1),
          coeff[:,1]*torch.exp(-(torch.sum(vy[...,1] - adv_v[...,0], dim=(1,2))**2)).unsqueeze(1)), dim=1).norm(dim=1).unsqueeze(1)
        

    def _ns_conservation(self, vx, vy, dt, dx, coeff):
        '''
            Calculated 2nd order central difference scheme for nonlinear advection operator
        '''
        raise NotImplementedError
        #un = vx.clone()
        #vn = vy.clone()
        #con_u = torch.empty(vx.shape)
        #con_v = torch.empty(vy.shape)
        #for idx in range(vx.shape[0]):
        #    cx = coeff[idx][0]
        #    cy = coeff[idx][1]

        #    con_u[idx] = torch.roll(un[idx], shifts=(-1), dims=0) - torch.roll(un[idx], shifts=(1), dims=(0))
        #    con_v[idx] = torch.roll(vn[idx], shifts=(-1), dims=1) - torch.roll(vn[idx], shifts=(1), dims=(1))

        # Calculate update
        #con_u = con_u/(2*dx)
        #con_v = con_v/(2*dx)

        #conserve = con_u + con_v

        #print(conserve.shape)
        #print(conserve.sum(dim=(1,2,3)))
        #raise

        #return torch.exp(-(torch.sum(vx[...,1] - adv_u[...,0], dim=(1,2))**2)).unsqueeze(1),
          #coeff[:,1]*torch.exp(-(torch.sum(vy[...,1] - adv_v[...,0], dim=(1,2))**2)).unsqueeze(1)), dim=1).norm(dim=1).unsqueeze(1)


    def _pde_similarity(self, u, dt, dx, target):
        
        # Only do this to NS data
        mask = target[:,-1] == 1

        # Similarity from non-NS data
        u_diff = self._diffusion(u[~mask], dt[~mask], dx, target[~mask][:,0].unsqueeze(-1))
        u_advection = self._advection(u[~mask], dt[~mask], dx, target[~mask][:,1:3].unsqueeze(-1))
        sim_weights = torch.cat((u_advection, u_diff, torch.zeros(u_diff.shape)), dim=1)

        # Similarity weights from NS data
        vx, vy = self._convert_ns(u[mask]) # Convert from vorticity to velocity
        u_diff_ns = self._ns_diffusion(vx, vy, dt[mask], dx, target[mask][:,0].unsqueeze(-1))
        u_advection_ns = self._ns_advection(vx, vy, dt[mask], dx, target[mask][:,1:3].unsqueeze(-1))
        #u_conservation_ns = self._ns_conservation(vx, vy, dt[mask], dx, target[mask][:,-1].unsqueeze(-1))
        u_conservation_ns = torch.ones(u_diff_ns.shape)
        sim_weights_ns = torch.cat((u_advection_ns, u_diff_ns, u_conservation_ns), dim=1)

        # Assemble 
        xi = torch.zeros((mask.shape[0], 3))
        xi[~mask] = sim_weights
        xi[mask] = sim_weights_ns
        xj = xi.clone()
        #print(xj.shape)
        #print(xj)
        #raise

        # Calculate matrix of dot products
        prod_mat = torch.sqrt(torch.sum((xi.unsqueeze(0) * xj.unsqueeze(1)).abs(), dim=-1))
        
        # Calculate matrix of maximum magnitudes
        norm_vec = torch.max(torch.cat((xi.norm(dim=-1).unsqueeze(-1), xj.norm(dim=-1).unsqueeze(-1)), dim=-1), dim=-1)[0]
        norm_mat1 = torch.ones(xi.shape[0]).unsqueeze(0) * norm_vec.unsqueeze(1) 
        norm_mat2 = norm_vec.unsqueeze(0) * torch.ones(xi.shape[0]).unsqueeze(1)
        norm_mat = torch.cat((norm_mat1.unsqueeze(-1), norm_mat2.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return prod_mat / norm_mat


    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _similarity(self, vi, vj):
        return torch.sqrt(vi.dot(vj).abs())/torch.max(vi.norm(), vj.norm())


    def _convert_ns(self, u):

        ###
        #  Taken from FNO simulation code
        ###

        #Grid size - must be power of 2
        N = u.size()[-2]
   
        #Maximum frequency
        k_max = math.floor(N/2.0)

        #Wavenumbers in y-direction
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                         torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).repeat(N,1)
        #Wavenumbers in x-direction
        k_x = k_y.transpose(0,1)
  
        #Truncate redundant modes
        k_x = k_x[..., :k_max + 1]
        k_y = k_y[..., :k_max + 1]
  
        #Negative Laplacian in Fourier space
        lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
        lap[0,0] = 1.0

        #Initial vorticity to Fourier space
        w_h = torch.fft.rfft2(u, dim=(-3,-2))
        psi_h = w_h / lap.view(1,lap.shape[0],lap.shape[1],1)

        #Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y.view(1, lap.shape[0], lap.shape[1], 1) * 1j * psi_h
        qx = q.clone()
        q = torch.fft.irfft2(q, s=(N, N), dim=(-3,-2))

        #Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x.view(1, lap.shape[0], lap.shape[1], 1) * 1j * psi_h
        qy = v.clone()
        v = torch.fft.irfft2(v, s=(N, N), dim=(-3,-2))

        return q, v


    def forward(self, x1, x2, vs, u, dt, dx):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        sim_score = np.zeros((batch_size, batch_size-1))

        #print("\nHERE\n")
        #print(u.shape, dt.shape, dx.shape, vs.shape)
        #print(vs)
        #print(u.shape, dt.shape, dx.shape, vs.shape)
        sim = self._pde_similarity(u, dt, dx, vs)
        #print(sim.shape)
        #raise

        #print(sim)
        #print(sim_score.shape)
        #print(sim_score)

        #sim_score = 1 - self.lambda_1 * torch.tensor(sim, dtype=torch.float).to(x1.device)
        sim_score = 1 - self.lambda_1 * sim
        sim_score = sim_score.repeat(2, 2)
        sim_score = torch.nan_to_num(sim_score, nan=0.)
        #print(sim_score.shape)
        #print(sim_score)
        #raise

        representations = torch.cat([x2, x1], dim=0)

        ###
        # TODO: Double check this is doing what I want
        ###
        similarity_matrix = self.similarity_function(representations[...,0],
                                                     representations[...,0])

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        negatives *= sim_score[mask_samples_from_same_repr].view(2*batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        if(loss.isnan()):
            print(similarity_matrix)
            print(sim_score)
            print(logits)

        return loss #/ (2 * batch_size)


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask
    
    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N - 1) augmented examples within a minib    atch as negative examples.
        """ 
        N = 2 * self.batch_size * self.world_size
        
        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)
        
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too
        # resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

 
class SimCLR(nn.Module):
    def __init__(self, encoder, h_size, z_size, variables=None):

        super(SimCLR, self).__init__()

        '''
        input size: batch_size * 25 * 25 * 1
        encoder output size: batch_size * h_size
        '''

        self.h_size = h_size
        self.z_size = z_size

        self.encoder = encoder
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.h_size, self.h_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.h_size, z_size, bias=False),
        )
        
    def forward(self, x_i, x_j, dt, variables=None):
        num = x_i.shape[0]

        # Spatial information
        if(len(x_i.shape) == 3):
            grid_i = x_i[:,-1,:]
            grid_j = x_j[:,-1,:]
            h_i = self.encoder(x_i[:,:-1,:], grid_i, dt, variables)#.reshape(num, -1)
            h_j = self.encoder(x_j[:,:-1,:], grid_j, dt, variables)#.reshape(num, -1)
        elif(len(x_i.shape) == 4):
            grid_i = x_i[:,-2:]
            grid_j = x_j[:,-2:]
            #h_i = self.encoder(x_i[:,:-2], grid_i, dt, variables)#.reshape(num, -1)
            #h_j = self.encoder(x_j[:,:-2], grid_j, dt, variables)#.reshape(num, -1)
            h_i = self.encoder(x_i[:,:-2], grid_i)
            h_j = self.encoder(x_j[:,:-2], grid_j)

        z_i = self.projector(h_i).reshape(num, -1)
        z_j = self.projector(h_j).reshape(num, -1)
        return h_i, h_j, z_i, z_j


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

