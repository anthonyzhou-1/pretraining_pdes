import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
import torch 
import numpy as np

def vis_1D(traj, ax=None, title=""):
    '''
    Plots 1D PDE trajectory, with lighter colors for later time steps
    Traj is expected in shape: [nt, nx]
    '''
    N = traj.shape[0]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N)))
    x = torch.linspace(0, 2, traj.shape[1])
    for i in range(N):
        ax.plot(x, traj[i])
    ax.set_title(title)

def vis_1D_im(u, ax = None, fig=None, title='test', aspect='auto', cmap='viridis', vmin=None, vmax=None):
    '''
    Plots 1D PDE trajectory as an image
    Traj is expected in shape: [nt, nx]
    '''
    im = ax.imshow(u, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)  
    ax.set_title(title)
    fig.colorbar(im, ax=ax)


def vis_2d(u, title="", cmap = "seismic", vmin = None, vmax = None):
    '''
    Makes gif of 2D PDE trajectory
    u is expected in shape [nt, nx, ny]
    '''

    # Initialize plot
    gs = GridSpec(1, 2, width_ratios = [0.9, 0.05])
    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(gs[0], projection = '3d', proj_type = 'ortho')
    cbar_ax = fig.add_subplot(gs[1])

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    ax.set_title(title)
    for i in range(u.shape[0]):
        im = ax.imshow(u[i].squeeze(), animated=True, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, cax=cbar_ax)
        ims.append([im])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save(f"assets/{title}.gif", writer=writer)
    print("saved")

def vis_2d_subplots(data, title="", cmap = "seismic", minmin= None, maxmax = None):
    u, u_masked, u_rec = data

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5), width_ratios=[5, 5, 5, 1])
    frames = [] # store generated images
    for i in range(len(u)):
        
        img1 = ax1.imshow(u[i], vmin=minmin, vmax=maxmax, cmap=cmap, animated=True)
        ax1.set_title('original')
        img2 = ax2.imshow(u_masked[i], vmin=minmin, vmax=maxmax, cmap=cmap, animated=True)
        ax2.set_title('masked')
        img3 = ax3.imshow(u_rec[i], vmin=minmin, vmax=maxmax, cmap=cmap, animated=True)
        ax3.set_title('reconstructed')
        cbar = fig.colorbar(img3, cax=ax4)
        frames.append([img1, img2, img3])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save(f"assets/{title}.gif", writer=writer)
    print('saved at: ', f'assets/{title}.gif')

def vis_2d_plot(u, grid, title="", downsample=1):
    '''
    Plots 2D PDE trajectory on a 3D plot as an mp4
    u is expected in shape [nt, nx, ny]
    grid is expected in shape [2, nx, ny]
    '''

    norm = colors.Normalize()
    X, Y = grid
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    if downsample > 1:
        X = X[::downsample, ::downsample]
        Y = Y[::downsample, ::downsample]
        u = u[::downsample, ::downsample, ::downsample]
    nt, nx, ny = u.shape

    def animate(n):
        ax.cla()

        x = u[n]
        cmap = cm.jet(norm(x))
        ax.plot_surface(X, Y, x, facecolors=cmap, rstride=1, cstride=1)
        ax.set_zlim(-1.5, 1.5)

        return fig,


    anim = FuncAnimation(fig = fig, func = animate, frames = nt, interval = 1, repeat = False)
    anim.save(f'assets/{title}.mp4',fps=10)

def vis_2d_surfaces(data, grid, title="", cmap = "seismic"):
    u, u_masked, u_rec = data

    nt = len(u)

    gs = GridSpec(1, 4, width_ratios = [0.9, 0.9, 0.9, 0.05])
    fig = plt.figure(figsize = (16, 5))
    ax1 = fig.add_subplot(gs[0], projection = '3d', proj_type = 'ortho')
    ax2 = fig.add_subplot(gs[1], projection = '3d', proj_type = 'ortho')
    ax3 = fig.add_subplot(gs[2], projection = '3d', proj_type = 'ortho')
    cbar_ax = fig.add_subplot(gs[3])

    norm = colors.Normalize()
    X, Y = grid

    def myPlot(ax, X, Y, U):
        cmap = cm.jet(norm(U))
        surf = ax.plot_surface(X, Y, U, facecolors = cmap,
                            rstride=1, cstride=1, antialiased = False)
        ax.set_zlim(-1.5, 1.5)

        fig.colorbar(surf, shrink = 0.5, aspect = 5, cax = cbar_ax)

    def animate(i):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.set_title('original')
        ax2.set_title('masked')
        ax3.set_title('reconstructed')
        myPlot(ax1, X, Y, u[i])
        myPlot(ax2, X, Y, u_masked[i])
        myPlot(ax3, X, Y, u_rec[i])

        

    anim = FuncAnimation(fig = fig, func = animate, frames = nt, interval = 1, repeat = False)
    anim.save(f'assets/{title}.mp4',fps=10)