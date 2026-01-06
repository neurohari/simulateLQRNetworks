# this file generates random networks with different connection densities and saves them to a HDF5 file.
# 

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from lib import spectral_abscissa, Gramians
from plotlib import plot_eig, load_plot_setting

plt.ion()  # Turn on interactive mode

def init_connectivity(p, w0, n):
    W = np.random.lognormal(np.log(w0),1,size=[n,n])
    #generate a 1's and 0's maxtrix with density p
    adjacency = np.random.uniform(0,1,size=[n,n])
    adjacency[adjacency < p] = 1 # Values less than p are set to 1
    adjacency[adjacency >= p] = 0 # Values greater than or equal to p are set to 0
    W = W * adjacency
    W = W - np.diag(np.diag(W)) # remove self connections
    return W

def generate_random_network(p):
    n = 100 # number of neurons
    w0 = 1 # excitatory synaptic strength
    W = np.random.lognormal(np.log(w0),1,size=[n,n])
    #generate a 1's and 0's maxtrix with density p
    adjacency = np.random.uniform(0,1,size=[n,n])
    adjacency[adjacency < p] = 1 # Values less than p are set to 1
    adjacency[adjacency >= p] = 0 # Values greater than or equal to p are set to 0
    W = W * adjacency
    W = W - np.diag(np.diag(W)) # remove self connections
    return W

def generate_random_networks(p_values, seed):
    np.random.seed(seed)

    networks_by_pvals = {}
    for p in p_values:
        networks = []
        for i in range(10):
            W = generate_random_network(p)
            networks.append(W)
        networks_by_pvals[p] = networks
    return networks_by_pvals

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

def load_networks_by_pvals(filename, p):
    networks = []
    with h5py.File(filename, 'r') as f:
        p_group = str(p)  # Ensure the p value is in the correct format
        if p_group in f:
            for i in range(10):
                dataset_name = f'{p_group}/network_{i}'
                if dataset_name in f:
                    network = f[dataset_name][()]
                    networks.append(network)
                else:
                    print(f"Dataset {dataset_name} not found.")
        else:
            print(f"p group {p_group} not found.")
    return networks

def main():
    parser = argparse.ArgumentParser(description="Generate random networks with different connection densities and save them to a HDF5 file.")
    parser.add_argument("--filename", type=str, default='density_networks.hdf5', help="The name of the HDF5 file to save the networks to.")
    parser.add_argument("--p_values", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8], help="The connection densities to generate networks for.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use for generating random networks.")
    args = parser.parse_args()

    args.filename = f'C:/Users/kalidindi/OneDrive - UCL/Documents/GitHub/FNRSmain/OptRNNctrl_ISN_MPC/datastore/WeightsData/Density/{args.filename}'

    networks_by_pvals = generate_random_networks(args.p_values, args.seed)
    with h5py.File(args.filename, 'w') as f:
        for p, networks in networks_by_pvals.items():
            group = f.create_group(str(p))
            for i, network in enumerate(networks):
                group.create_dataset(f'network_{i}', data=network)

    # Visualize the data in the HDF5 file
    print_hdf5_structure(f'{args.filename}')
    
    # load networks by spectral radius
    p = 0.2  # Example connection density
    networks = load_networks_by_pvals(args.filename, p)
    print(f"Loaded {len(networks)} networks for spectral radius {p}.")

if __name__ == "__main__":
    main()
