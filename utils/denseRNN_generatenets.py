# This file generates the ISN neural networks with different spectral radiuses
# and saves them to a HDF5 file.

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from lib import spectral_abscissa, Gramians 
from plotlib import plot_eig, load_plot_setting
import os
from scipy.integrate import odeint


plt.ion()  # Turn on interactive mode

def init_connectivity(n_e, n_i, p, w0, e, i, n):
    W = np.random.lognormal(np.log(w0),1,size=[n,n])
    #build adjacency matrix of connections
    adjacency = np.random.uniform(0,1,size=[n,n_e])
    adjacency[adjacency>=p] = 2 # what is the point of 2 here?
    adjacency[adjacency<p] = 1
    adjacency[adjacency>1] = 0
    W[e] = W[e] * adjacency # remove silence percent of the synapses
    W[i] = -np.random.uniform(0,1,size=[n,n_i])
    W = W - np.diag(np.diag(W)) # remove self connections 
    return W


def normalise(W, i, n_i):
    dc_eval = -10 # eigenvalue of DC mode
    W[i] = np.minimum(0, W[i])
    z = (dc_eval - np.sum(W, axis=1).reshape(-1,1)) / n_i
    W[i] = np.minimum(0, z + W[i])
    W = W - np.diag(np.diag(W))
    return W


def generate_isn_network(spectral_radius):
    n_e = 80 # number of excitatory neurons
    n_i = 20  # number of inhibitory neurons
    n = n_e + n_i  # number of neurons
    p = 0.2 # excitatory connection density
    w0 = spectral_radius / np.sqrt(p * (1-p) * n) # excitatory synaptic strength 

    # useful slicing functions to extract the relevant blocks of the connectivity matrix
    e = slice(0,None),slice(0,n_e)
    i = slice(0,None),slice(n_e,None)
    ee = slice(0,n_e), slice(0,n_e)
    ei = slice(0,n_e), slice(n_e,None)
    ie = slice(n_e,None), slice(0,n_e)
    ii = slice(n_e,None), slice(n_e,None)

    # initialise connectivity matrix with the right density
    W = init_connectivity(n_e, n_i, p, w0, e, i, n)

    # normalize the connectivity
    W = normalise(W, i, n_i)
    

    # stability optimization    
    # optimisation parameters
    n_iter = 5000 # number of training steps
    threshold = 0.8 # stop optimization when spectral abscissa lower than threshold 
    eta = 1  # initial learning rate
    # optimisation flags
    print_every = 300
    plot_every = 300 
    #plt.ion()
    # start stability optimization (takes about 1000 iterations)
    for step in range(n_iter):
        s = spectral_abscissa(W) 
        if s < threshold: 
            print("FINAL spectral abscissa  %f" % s)
            W_evals, _ = np.linalg.eig(W)
            break
        # adaptive learning rate
        eta = min(max(0.5 * eta if s > s_ else 1.01 * eta,1),8) if step > 0 else 1
        s_ = s # copy of spectral abscissa in this iteration for updating of learning rate in next iteration
        shift = max(1, 1.2*s) 
        A = W - shift * np.eye(n)
        Q = Gramians.obsv(A) # observability gramian
        P = Gramians.ctrl(A) # controllability gramian
        QP = Q.dot(P)
        dW = QP / np.trace(QP) # gradient of smoothed spectral abscissa with respect to W
        W[i]  = W[i] - eta * dW[i]
        W = normalise(W, i, n_i) # normalise W after each iteration
        
    # convert the continuous network to discrete one with tau
    tau = 20e-3
    N = W.shape[0]
    W = (W - np.eye(N)) / tau

    return W

def generate_dense_RNN(g):
    n = 100 # number of units in the network
    # generate a random RNN with a given variance of the weights
    W = np.random.normal(0, g/np.sqrt(n), (n,n))
    # convert the continuous network to discrete one with tau
    #W = W - np.diag(np.diag(W)) # remove the self connections
    tau = 20e-3
    N = W.shape[0]
    W = (W - np.eye(N)) / tau
    return W

def generate_isn_networks(spectral_radiuses, seed):
    np.random.seed(seed)
    # each spectral radius has 10 networks. Each of 10 networks should be nested in a list in the hdf5 file]
    networks_by_radius = {}
    for spectral_radius in spectral_radiuses:
        networks = []
        for i in range(10):
            networks.append(generate_isn_network(spectral_radius))
        networks_by_radius[spectral_radius] = networks
    return networks_by_radius

def generate_dense_RNNs(spectral_radiuses, seed):
    np.random.seed(seed)
    # each spectral radius has 10 networks. Each of 10 networks should be nested in a list in the hdf5 file]
    networks_by_radius = {}
    for spectral_radius in spectral_radiuses:
        networks = []
        for i in range(10):
            networks.append(generate_dense_RNN(spectral_radius))
        networks_by_radius[spectral_radius] = networks
    return networks_by_radius

def save_isn_networks(networks_data, readouts_data, args):
    """
    Save networks and readouts data into an HDF5 file, grouped by spectral radius.

    :param filename: Name of the HDF5 file to save.
    :param spectral_radii: List of spectral radii.
    :param networks_data: Dictionary with spectral radius as keys and list of networks as values.
    :param readouts_data: Dictionary with spectral radius as keys and list of readouts as values.
    """
    spectral_radii = args.spectral_radiuses
    filename = args.filename
    with h5py.File(filename, 'w') as file:
        for radius in spectral_radii:
            # Create a group for each spectral radius
            radius_group = file.create_group(str(radius))
            
            # Create subgroups for networks and readouts
            networks_group = radius_group.create_group('networks')
            readouts_group = radius_group.create_group('readouts')
            
            # Assuming networks_data and readouts_data are dictionaries containing
            # lists of networks and readouts for each spectral radius, respectively
            for i, network in enumerate(networks_data[radius]):
                networks_group.create_dataset(f'network_{i}', data=network)
                
            for i, readout in enumerate(readouts_data[radius]):
                readouts_group.create_dataset(f'readout_{i}', data=readout)

def save_dense_RNNs(networks_data, readouts_data, args):
    spectral_radii = args.spectral_radiuses
    filename = args.filename
    with h5py.File(filename, 'w') as file:
        for radius in spectral_radii:
            # Create a group for each spectral radius
            radius_group = file.create_group(str(radius))
            
            # Create subgroups for networks and readouts
            networks_group = radius_group.create_group('networks')
            readouts_group = radius_group.create_group('readouts')
            
            # Assuming networks_data and readouts_data are dictionaries containing
            # lists of networks and readouts for each spectral radius, respectively
            for i, network in enumerate(networks_data[radius]):
                networks_group.create_dataset(f'network_{i}', data=network)
                
            for i, readout in enumerate(readouts_data[radius]):
                readouts_group.create_dataset(f'readout_{i}', data=readout)

def load_networks_by_spectral_radius(filename, spectral_radius):
    networks = []
    with h5py.File(filename, 'r') as f:
        spectral_radius_group = str(spectral_radius)  # Ensure the spectral radius is in the correct format
        if spectral_radius_group in f:
            for i in range(10):  # Assuming there are always 10 networks
                dataset_name = f'{spectral_radius_group}/networks/network_{i}'
                if dataset_name in f:
                    network = f[dataset_name][()]
                    networks.append(network)
                else:
                    print(f"Dataset {dataset_name} not found.")
        else:
            print(f"Spectral radius group {spectral_radius_group} not found.")
    return networks

def load_dense_RNNs_by_spectral_radius(filename, spectral_radius):
    networks = []
    with h5py.File (filename, 'r') as f:
        spectral_radius_group = str(spectral_radius)
        if spectral_radius_group in f:
            for i in range(10):
                dataset_name = f'{spectral_radius_group}/networks/network_{i}'
                if dataset_name in f:
                    network = f[dataset_name][()]
                    networks.append(network)
                else:
                    print(f"Dataset {dataset_name} not found.")
        else:
            print(f"Spectral radius group {spectral_radius_group} not found.")
    return networks


def print_hdf5_structure(filename):
    """
    Print the structure of an HDF5 file, including groups and datasets.

    :param filename: Path to the HDF5 file.
    """
    # Print the absolute path of the file
    absolute_path = os.path.abspath(filename)
    print(f"Reading HDF5 file from: {absolute_path}")
    def print_structure(name, obj):
        """
        Recursive function to print the structure of groups and datasets.
        
        :param name: Name of the current object.
        :param obj: HDF5 group or dataset object.
        """
        print(name)
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
        else:
            print(f"Unknown: {name}")
    
    with h5py.File(filename, 'r') as file:
        file.visititems(print_structure)#

# compute observability grammian of all networks in a for loop 
from scipy.linalg import solve_continuous_lyapunov
def compute_controllability_grammian(A):
    B = np.eye(A.shape[0])
    Q = B @ B.T
    W = solve_continuous_lyapunov(A, Q)
    return W

def main():
    parser = argparse.ArgumentParser(description='Generate ISN neural networks with different spectral radiuses and save them to a HDF5 file.')
    parser.add_argument('--filename', type=str, default = 'dense_network_weights.hdf5', help='The HDF5 file to save the networks to.')
    parser.add_argument('--spectral_radiuses', type=float, nargs='+', default=[0.4, 0.6, 0.7, 0.8, 1.1], help='The spectral radiuses of the networks.')
    parser.add_argument('--seed', type=int, default=45, help='The seed for the random number generator.')
    args = parser.parse_args()

    np.random.seed(args.seed)
    # the file should be in the data store directory above the current directory
    args.filename = f'././datastore/WeightsData/denseRNN/{args.filename}'

    
    networks_by_radius = generate_dense_RNNs(args.spectral_radiuses, args.seed)

    S = 8  # number of states
    N = 100  # number of neurons
    readouts_by_radius = {}
    wout = np.zeros((S, N))  # (states X neuron dim)
    # only the rwos 2 and 6 of Wout should be filled with normal random numbers (for fx and fy actuation states)
    #wout[2, :] = np.random.normal(0, 0.05/np.sqrt(1/N), (1, N))
    #wout[6, :] = np.random.normal(0, 0.05/np.sqrt(1/N), (1, N))
    wout[2, :] = np.random.normal(0, 0.5, (1, N))
    wout[6, :] = np.random.normal(0, 0.5, (1, N))
    for radius in args.spectral_radiuses:
        readouts = []
        for i in range(10):
            readouts.append(wout)
        readouts_by_radius[radius] = readouts
    save_dense_RNNs(networks_by_radius, readouts_by_radius, args)

    # loop over all networks
    gram_by_radius = {0.4: {}, 0.6: {}, 0.7: {}, 0.8: {}, 1.1: {}}
    for radius in args.spectral_radiuses:
        networks = networks_by_radius[radius]
        readouts = readouts_by_radius[radius]
        for i in range(10):
            network = networks[i]
            readout = readouts[i]
            # compute the observability gramman of each
            # network and readout pair
            Gram = compute_controllability_grammian(network)
            # compute the controllability as
            Contr = np.trace(readout @ Gram @ readout.T) / 2
            # collect grammian matrix at each spectral radius and each i
            # network
            gram_by_radius[radius][i] = Contr

    # plot the grammian at 0.2 spectral radius
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i in range(10):
        ax.plot(gram_by_radius[0.4][i], 'o', label='network %i' % (i+1))
    plt.show()

    # example network dynamic response
    example_wrec = networks_by_radius[args.spectral_radiuses[3]][0]
    example_wout = readouts_by_radius[args.spectral_radiuses[3]][0]
    n, _ = np.shape(example_wrec)

    tau = 20e-3
    W = example_wrec + 0
    A = (W + np.eye(n)) * tau
    # compute the gramians
    G = Gramians(A)
    # get the observability gramian OG
    OG = G.O
    # get the eigenvalues and eigenvectors of the observability gramian
    Q_evals = OG.evals
    Q_evecs = OG.modes

    plt.figure(figsize=(5,5))
    plt.ylabel("energy evoked $\epsilon$")
    plt.xlabel("$n^{th}$ eigenvector of $Q$")
    plt.plot(Q_evals, "k")
    plt.show()
    

    phi = lambda x: x # rate function 
    def f(x,t):
        return (W)@ x
    figsize=(20,4.5)
    tf = 0.5 # simulate for 500ms
    dt = 1e-2
    n_steps = int(tf/dt) + 1
    indices = [int(idx) for idx in [0, 0.05*n-1, 0.2*n-1, 0.4*n-1, 0.8*n-1]]

    fig, axes = plt.subplots(1,len(indices), figsize=figsize, sharey=True)
    axes[0].set_ylabel("firing rate [a.u.]")
    axes[0].set_yticks(np.arange(-5,5.1,3))
    ts = np.linspace(0,tf,n_steps)
    for i, m in enumerate(indices):
        x0 = Q_evecs[:,m]
        xs = odeint(f,x0,ts)
        rs = phi(xs) 
        axes[i].set_ylim(-.5,.5)
        axes[i].set_xlabel("t [s]")
        axes[i].set_xticks(np.arange(0,0.51,0.25))
        axes[i].text(0.35,0.9, "mode %i" % (m+1), transform=axes[i].transAxes, fontsize=20)
        idxs = range(0,n,10)
        colors=iter(plt.cm.RdBu_r(np.linspace(0,1,len(idxs))))
        for j in idxs:
            c=next(colors)
            axes[i].plot(ts,rs[:,j],c=c)
        
    plt.show(block=True)

    # compute eigen values of the network A
    W_evals, W_evecs = np.linalg.eig(20e-3*(W + np.eye(n)))
    # plot the eigenvalues of the network A on the complex plane
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.plot(W_evals.real, W_evals.imag, 'o')
    plt.show(block=True)


    


if __name__ == '__main__':
    main()