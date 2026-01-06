import numpy as np


# generate a dense network with no small-worldness & normal distribution of weights

def smallworldWeights_3Dskewsym(S, w):
    ''' N = neurons
        K = number of clusters
        S = size of state feedback
        cinp, coutp, c = probability of connections between a random pair
        c1, c2 = out group, in group connection probabilities between a pair of neurons
        sparse_affs = if the sensory and motor projections need to be sparse (i.e., separating S1 and M1 regions)'''
    Win = np.zeros((3, S))  # (neuron dim X state feedback dim)
    Wout = np.zeros((S, 3))  # (states X neuron dim)
    Wrec = np.zeros((3, 3))  # (neuron dim X neuron dim)

    # Set the adjacency matrix
    Wrec = np.array([[0, w, -w], [-w, 0, w], [w, -w, 0]])
    
    # generate fully-connected uniform distribution of weights for inp and outp for now
    # later tune these matrices only to form connection with one of the multiple clusters in the cortex
    # input weights must be connected to K = 1 and output must be drawn from K = 4 (in a 4 cluster cortex model).
    Win[:3, :] = 0*np.random.normal(0, 0.5/np.sqrt(1/S), (3, S))
    
    # only the rwos 2 and 6 of Wout should be filled with normal random numbers (for fx and fy actuation states)
    Wout[2, :] = np.random.normal(0, 1/np.sqrt(3), (1, 3))
    Wout[6, :] = np.random.normal(0, 1/np.sqrt(3), (1, 3))

    #Wout[2, :] = np.array([[0, 1, 1]])
    #Wout[6, :] = np.array([[1, 0, 1]])


    return [Win, Wrec, Wout]










# sparse network without any EI components, gaussian distribution of weights with different spectral radius.
def sparseNetwork_BSG(N, S, alpha = 0.1):
    ''' N = neurons
        S = size of state feedback
        cinp, coutp, c = probability of connections between a random pair
        c1, c2 = out group, in group connection probabilities between a pair of neurons
        sparse_affs = if the sensory and motor projections need to be sparse (i.e., separating S1 and M1 regions)'''
    Win = np.zeros((N, S))  # (neuron dim X state feedback dim)
    Wout = np.zeros((S, N))  # (states X neuron dim)
    Wrec = np.zeros((N, N))  # (neuron dim X neuron dim)

    Win[:, :] = 0*np.random.normal(0, 1/np.sqrt(1/S), (N, S))
    
    # only the rwos 2 and 6 of Wout should be filled with normal random numbers (for fx and fy actuation states)
    Wout[2, :] = np.random.normal(0, 1/np.sqrt(1/N), (1, N))
    Wout[6, :] = np.random.normal(0, 1/np.sqrt(1/N), (1, N))


    # generate recurrent connections
    weighted_connectome = np.zeros((N, N))
    '''For each neuron, select alpha*N/2 presynaptic excitatory neurons randomly and assign them weight Wexc.
    Also, select alpha*N/2 presynsptic inhibitory neurons randomly and assign them weight Winh'''
    for n in range(weighted_connectome.shape[0]):
        exc_connections = np.random.randint(low = 0, high = int(N/2), size = int(alpha*N / 2))
        inh_connections = np.random.randint(low = int(N/2), high = N, size = int(alpha*N / 2))
        #weighted_connectome[n, m] = np.random.normal(0, alpha / np.sqrt(N))
        weighted_connectome[n, exc_connections] = 1 / (np.sqrt(N * alpha * (1 - alpha)))
        weighted_connectome[n, inh_connections] = - weighted_connectome[n, exc_connections]
    Wrec = weighted_connectome + 0.0

    # generate a normal distribution of weights for the recurrent connections with variance 1/N
    Wrec = np.random.normal(0, 6/np.sqrt(N), (N, N))



    return [Win, Wrec, Wout]