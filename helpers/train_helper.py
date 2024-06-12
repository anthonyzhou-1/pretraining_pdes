import torch
import random
from torch.utils.data import DataLoader
from common.utils import DataCreator
import argparse

def pretraining_loop(
        args: argparse,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim,
        scheduler,
        loader: DataLoader,
        data_creator: DataCreator,
        loss_fn,
        device
        ) -> torch.Tensor:

    l2_full = 0
    validation = False
    difficulty = get_difficulty(epoch, args.pretraining_epochs * data_creator.t_res, args.difficulty_threshold)

    with torch.set_grad_enabled(not validation):
        for bn, (xx, grid, coeffs) in enumerate(loader):

            # Creates data at t
            batch_size = xx.shape[0]
            if(args.mode == 'next_step'): # Double check when using pushforward/temporal bundling
                steps = [t for t in range(data_creator.tw,
                      data_creator.t_res - data_creator.tf + 1)]
            elif(args.mode == 'fixed_future'):
                if args.mode_pt == "rand":
                    steps = [t for t in range(data_creator.tw,
                      args.base_resolution[0] + 1)]
                else:
                    steps = [data_creator.target]
            random_steps = random.choices(steps, k=batch_size)
            
            if args.mode_pt == "rand" and args.mode == 'fixed_future':
                x = data_creator.create_rand_data(xx, random_steps)
            else:
                x, labels = data_creator.create_data(xx, random_steps)

            # Each model handles input differently
            if(args.pretraining == 'picl'):
                y_pred = model(u=x, grid=grid, dt=torch.Tensor([loader.dataset.dt]), variables=coeffs)

                t = torch.Tensor([loader.dataset.t[1] - loader.dataset.t[0]]).to(device=x.device)
                t = t.broadcast_to(x.shape[0])
                
                # Stack equation coefficients
                if(len(xx.shape) == 3): # 1D
                    target = torch.cat((coeffs['alpha'].unsqueeze(-1),
                                        coeffs['beta'].unsqueeze(-1),
                                        coeffs['gamma'].unsqueeze(-1)), dim=-1).cuda()
                    y_pred = y_pred.transpose(1,2)
                    x = x.transpose(1,2)
                    dx = grid[0][1] - grid[0][0]
                elif(len(xx.shape) == 4): # 2D
                    target = torch.cat((
                        coeffs['nu'].unsqueeze(-1),  # Diffusion
                        coeffs['ax'].unsqueeze(-1),  # Linear Advection
                        coeffs['ay'].unsqueeze(-1),  # Linear Advection
                        coeffs['cx'].unsqueeze(-1),  # Nonlinear Advection
                        coeffs['cy'].unsqueeze(-1)), # Nonlinear Advection
                        dim=-1).cuda()
                    y_pred = y_pred.permute(0, 2, 3, 1)
                    x = x.permute(0, 2, 3, 1)
                    dx = grid[0][0][0][1] - grid[0][0][0][0]

                dx = torch.Tensor([dx]) if(len(dx.shape) == 0) else dx
                loss = loss_fn(y_pred.cuda(), y_pred.cuda(), target.cuda(), x.cuda(), t.cuda(), dx.cuda())
            
            elif(args.pretraining == 'oddoneout'):
                loss = model(u=x, grid=grid, dt=torch.Tensor([loader.dataset.dt]), variables=coeffs, difficulty=difficulty)
            else:
                loss = model(u=x, grid=grid, dt=torch.Tensor([loader.dataset.dt]), variables=coeffs)

            # Take step
            if(loss.requires_grad):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            l2_full += loss.item()
            
            if(isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)):
                scheduler.step()

        if(not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) and l2_full != 0):
            scheduler.step()
    
    l2_full /= len(loader)
    return l2_full, bn+1, model, optimizer, scheduler


def training_loop(
        model: torch.nn.Module,
        optimizer: torch.optim,
        scheduler,
        unrolling: list,
        loader: DataLoader,
        data_creator: DataCreator,
        criterion: torch.nn.modules.loss,
        device: torch.cuda.device="cpu"
        ) -> torch.Tensor:

    model.train()
    losses = []
    dt = torch.tensor([loader.dataset.dt])
    for (u_super, grid, variables) in loader:

        batch_size = u_super.shape[0]
        unrolled_steps = random.choice(unrolling)
        if(data_creator.mode == 'next_step'):
            steps = [t for t in range(data_creator.tw,
                  data_creator.t_res - data_creator.tf - (data_creator.tf * unrolled_steps) + 1)]
            batch_steps = random.choices(steps, k=batch_size)
        elif(data_creator.mode == 'fixed_future'):
            steps = [data_creator.target]
            batch_steps = steps * batch_size

        data, labels = data_creator.create_data(u_super, batch_steps)

        if data_creator.mode == "fixed_future":
            for i in range(data_creator.tw):
                assert not torch.allclose(labels[:, 0], data[:, i]), "Data and labels are the same!"
        
        # Unrolling of the equation which serves as input at the current step
        # This is the pushforward trick!!!
        with torch.no_grad():
            for _ in range(unrolled_steps):
                batch_steps = [rs + data_creator.tf for rs in batch_steps]
                _, labels = data_creator.create_data(u_super, batch_steps)

                pred = model(data.to(device), grid, dt, variables)
                data = torch.cat((data, pred), dim=1)[:,-data_creator.tw:,...]

      
        pred = model(data.to(device), grid.to(device), dt.to(device), variables)
        loss = criterion(pred, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach() / batch_size)

        if(isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)):
            scheduler.step()

    if(not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)):
        scheduler.step()

    losses = torch.stack(losses)
    return torch.mean(losses)

def test_onestep_losses(model: torch.nn.Module,
                         loader: DataLoader,
                         data_creator: DataCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu",
                         verbose = False) -> None:
    # Set models to eval mode
    model.eval()
    dt = torch.tensor([loader.dataset.dt])
    steps = [data_creator.target]
    losses = []

    with torch.no_grad():
        for (u_super, grid, variables) in loader:
            batch_size = u_super.shape[0]
            steps = [data_creator.target]

            random_steps = random.choices(steps, k=batch_size)
            data, labels = data_creator.create_data(u_super, random_steps) 

            pred = model(data.to(device), grid, dt, variables)
                  
            loss = criterion(pred, labels.to(device))
            losses.append(loss / batch_size)

    losses = torch.stack(losses)
    if verbose:
        print(f'mean loss {torch.mean(losses)}')

    return torch.mean(losses)

def test_unrolled_losses(
        model: torch.nn.Module,
        nr_gt_steps: int,
        loader: DataLoader,
        data_creator: DataCreator,
        criterion: torch.nn.modules.loss,
        device: torch.cuda.device = "cpu",
        verbose = False
        ) -> torch.Tensor:

    losses = []
    model.eval()

    # Loop over every data sample
    dt = torch.tensor([loader.dataset.dt])

    all_tmp_losses = []
    for bn, (u_super, grid, variables) in enumerate(loader):
        batch_size = u_super.shape[0]
        losses_tmp = []
        with torch.no_grad():
            same_steps = [data_creator.tw * (nr_gt_steps+1)] * batch_size # Modified for batch size 1
            data, labels = data_creator.create_data(u_super, same_steps)
            
            pred = model(data.to(device), grid, dt, variables)
            loss = criterion(pred, labels.to(device)) 

            data = torch.cat((data, pred), dim=1)[:,-data_creator.tw:,...]
            losses_tmp.append(loss / batch_size)

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(data_creator.tw * (nr_gt_steps + 1), data_creator.t_res - data_creator.tf + 1, data_creator.tf):
                same_steps = [step] * batch_size
                _, labels = data_creator.create_data(u_super, same_steps) 
                
                pred = model(data.to(device), grid, dt, variables)
                loss = criterion(pred, labels.to(device))
                data = torch.cat((data, pred), dim=1)[:,-data_creator.tw:,...]
                
                losses_tmp.append(loss / batch_size)

        losses.append(torch.sum(torch.stack(losses_tmp)))

        all_tmp_losses.append(torch.stack(losses_tmp))

    all_tmp_losses = torch.stack(all_tmp_losses, dim=0)

    losses = torch.stack(losses)
    if verbose:
        print(f'Unrolled forward losses {torch.mean(losses)}')

    return torch.mean(losses)


def get_difficulty(current_epoch, max_epoch, difficulty_threshold):
   return current_epoch/max_epoch * difficulty_threshold
