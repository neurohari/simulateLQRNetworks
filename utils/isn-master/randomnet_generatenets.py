# this file generates random networks with different g / sqrt(N) where N is the number of neurons and g is the scaling factor of the connectivity matrix.
# and saves it to a HDF5 file.

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from lib import spectral_abscissa, Gramians
from plotlib import plot_eig, load_plot_setting

plt.ion()  # Turn on interactive mode

def init_connectivity(g, n):
    # the connectivity should be normal distributed with mean 0 and variance g^2/N
    W = np.random.normal(0,g/np.sqrt(n),size=[n,n])
    W = W - np.diag(np.diag(W)) # remove self connections
    return W

def generate_random_network(g):
    n = 100 # number of neurons
    # the connectivity should be normal distributed with mean 0 and variance g^2/N
    W = np.random.normal(0,g/np.sqrt(n),size=[n,n])
    W = W - np.diag(np.diag(W)) # remove self connections
    return W

def generate_random_networks(g_values, seed):
    np.random.seed(seed)

    # each g_value has 10 networks. Each of 10 networks should be nested in a list in the hdf5 file
    networks_by_gvals = {}
    for g in g_values:
        networks = []
        for i in range(10):
            W = generate_random_network(g)
            networks.append(W)
        networks_by_gvals[g] = networks
    return networks_by_gvals

def print_hdf5_structure(filename):
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name} (Dataset)")
        else:  # it's a Group
            print(f"{name} (Group)")
            for key in obj.keys():
                print_structure(f"{name}/{key}", obj[key])

    with h5py.File(filename, 'r') as f:
        print_structure('/', f)

def load_networks_by_gvals(filename, g):
    networks = []
    with h5py.File(filename, 'r') as f:
        g_group = str(g)  # Ensure the g value is in the correct format
        if g_group in f:
            for i in range(10):  # Assuming there are always 10 networks
                dataset_name = f'{g_group}/network_{i}'
                if dataset_name in f:
                    network = f[dataset_name][()]
                    networks.append(network)
                else:
                    print(f"Dataset {dataset_name} not found.")
        else:
            print(f"g group {g_group} not found.")
    return networks

def main():
    parser = argparse.ArgumentParser(description='Generate random networks with different g values.')
    parser.add_argument('--filename', type=str, default = 'gaussian_networks.hdf5', help='The HDF5 file to save the networks to.')
    parser.add_argument('--g_values', type=float, nargs='+', default=[0.0, 0.3, 0.6, 0.9, 1.2], help='The variances of the network connectivity.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    args.filename = f'C:/Users/kalidindi/OneDrive - UCL/Documents/GitHub/FNRSmain/OptRNNctrl_ISN_MPC/datastore/WeightsData/Gaussian/{args.filename}'


    networks_by_gvals = generate_random_networks(args.g_values, args.seed)

    with h5py.File(args.filename, 'w') as f:
        for g, networks in networks_by_gvals.items():
            g_group = f.create_group(str(g))
            for i, W in enumerate(networks):
                g_group.create_dataset(f'network_{i}', data=W)

    # Visualize the data in the HDF5 file
    print_hdf5_structure(f'{args.filename}')
    
    # load networks by spectral radius
    g = 0.3  # Example spectral radius
    networks = load_networks_by_gvals(args.filename, g)
    print(f"Loaded {len(networks)} networks for spectral radius {g}.")




if __name__ == '__main__':
    main()




