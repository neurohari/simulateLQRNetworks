import numpy as np
import scipy


# defining a function to set up cost matrices for the LQG controller


def setCostmatrices(Atot, Btot, delta, r, w):
    '''
    A,B,R,Q = setupMatrice(A0,B0,delta,w)
    Takes as input the matrices of continuous timme differential equation, 
    timestep and weights for the definition of the cost matrices
    
    Outputs: discrete time matrices and cost matrices
    '''
    ns = Atot.shape[0]
    nc = Btot.shape[1]
    print(ns)
    #Setup cost matrices
    nStep = w.shape[1] - 1
    # Control cost
    R = np.zeros(shape=(nStep, nc, nc))
    for i in range(nStep):
        R[i, :, :] = r*np.identity(nc)
    # State cost
    Q = np.zeros(shape=(nStep+1, ns, ns))
    vec = np.zeros(shape=(ns, 1))
    id = np.identity(int(ns/2))
    for i in range(nStep+1):
        for j in range(int(ns/2)):
            id_temp = id[:, j]
            vec[0:int(ns/2), 0] = np.transpose(id_temp) # Q matrix for the actual hand state
            vec[int(ns/2):ns, 0] = -np.transpose(id_temp) # Q entries for the target state (negative)
            #Q[i, :, :] = Q[i, :, :] + ((i/nStep)**25) * w[j, i]*vec @ np.transposse(vec)
            Q[i, :, :] = Q[i, :, :] + w[j, i]*vec @ np.transpose(vec)

    return R, Q



def setCostmatrices_wprep(Atot, Btot, delta, r, w, wprep, nPrep):
    '''
    A,B,R,Q = setupMatrice(A0,B0,delta,w)
    Takes as input the matrices of continuous timme differential equation, 
    timestep and weights for the definition of the cost matrices
    
    Outputs: discrete time matrices and cost matrices
    '''
    ns = Atot.shape[0]
    nc = Btot.shape[1]

    #Setup cost matrices
    nStep = w.shape[1] - 1
    nHold = 20
    # Control cost
    R = np.zeros(shape=(nStep, nc, nc))
    for i in range(nStep):
        R[i, :, :] = r*np.identity(nc)
    # State cost
    Q = np.zeros(shape=(nStep+1, int(3*ns/3), int(3*ns/3)))
    vec = np.zeros(shape=(int(3*ns/3), 1))
    id = np.identity(int(ns/3))
    for i in range(nPrep, nStep - nHold +1):
        for j in range(int(ns/3)):
            id_temp = id[:, j]
            vec[0:int(ns/3), 0] = np.transpose(id_temp) # Q matrix for the actual hand state
            vec[int(ns/3):int(2*ns/3), 0] = -np.transpose(id_temp) # Q entries for the target state (negative)
            #Q[i, :, :] = Q[i, :, :] + ((i/nStep)**25) * w[j, i]*vec @ np.transpose(vec)
            #print((w[j, i]*vec @ np.transpose(vec)).shape)
            Q[i, :, :] = Q[i, :, :] + w[j, i] * vec @ np.transpose(vec) 

    
    for i in range(nStep - nHold +1, nStep+1):
        for j in range(int(ns/3)):
            id_temp = id[:, j]
            vec[0:int(ns/3), 0] = np.transpose(id_temp) # Q matrix for the actual hand state
            vec[int(ns/3):int(2*ns/3), 0] = -np.transpose(id_temp) # Q entries for the target state (negative)
            #Q[i, :, :] = Q[i, :, :] + ((i/nStep)**25) * w[j, i]*vec @ np.transpose(vec)
            #print((w[j, i]*vec @ np.transpose(vec)).shape)
            Q[i, :, :] = Q[i, :, :] +  w[j, i] * vec @ np.transpose(vec)

    # add the preparation cost
    tprep = int(nPrep) + 0
    vec = np.zeros(shape=(int(3*ns/3), 1))
    id = np.identity(int(ns/3))

    for i in range(nPrep+1):
        for j in range(int(ns/3)):
            id_temp = id[:, j]
            vec[0:int(ns/3), 0] = np.transpose(id_temp) # Q matrix for the actual hand state
            vec[2*int(ns/3):3*int(ns/3), 0] = -np.transpose(id_temp)
            Q[i, :, :] = Q[i, :, :] + wprep[j, tprep]*vec @ np.transpose(vec)

    return R, Q



def setCostmatrices_wprep_MPC(Atot, Btot, delta, r, w, wprep, nPrep):
    '''
    A,B,R,Q = setupMatrice(A0,B0,delta,w)
    Takes as input the matrices of continuous timme differential equation, 
    timestep and weights for the definition of the cost matrices
    
    Outputs: discrete time matrices and cost matrices
    '''
    ns = Atot.shape[0]
    nc = Btot.shape[1]

    #Setup cost matrices
    nStep = w.shape[1] - 1
    nHold = 20
    nPrep_MPC = 5
    # Control cost
    R = np.zeros(shape=(nStep, nc, nc))
    for i in range(nStep):
        R[i, :, :] = r*np.identity(nc)
    # State cost
    Q = np.zeros(shape=(nStep+1, int(3*ns/3), int(3*ns/3)))
    vec = np.zeros(shape=(int(3*ns/3), 1))
    id = np.identity(int(ns/3))
    for i in range(nPrep_MPC, nStep - nHold +1):
        for j in range(int(ns/3)):
            id_temp = id[:, j]
            vec[0:int(ns/3), 0] = np.transpose(id_temp) # Q matrix for the actual hand state
            vec[int(ns/3):int(2*ns/3), 0] = -np.transpose(id_temp) # Q entries for the target state (negative)
            #Q[i, :, :] = Q[i, :, :] + ((i/nStep)**25) * w[j, i]*vec @ np.transpose(vec)
            #print((w[j, i]*vec @ np.transpose(vec)).shape)
            Q[i, :, :] = Q[i, :, :] + w[j, i] * vec @ np.transpose(vec) 

    
    for i in range(nStep - nHold +1, nStep+1):
        for j in range(int(ns/3)):
            id_temp = id[:, j]
            vec[0:int(ns/3), 0] = np.transpose(id_temp) # Q matrix for the actual hand state
            vec[int(ns/3):int(2*ns/3), 0] = -np.transpose(id_temp) # Q entries for the target state (negative)
            #Q[i, :, :] = Q[i, :, :] + ((i/nStep)**25) * w[j, i]*vec @ np.transpose(vec)
            #print((w[j, i]*vec @ np.transpose(vec)).shape)
            Q[i, :, :] = Q[i, :, :] +  w[j, i] * vec @ np.transpose(vec)

    # add the preparation cost
    tprep_MPC = int(nPrep_MPC) + 0
    vec = np.zeros(shape=(int(3*ns/3), 1))
    id = np.identity(int(ns/3))

    for i in range(nPrep_MPC+1):
        for j in range(int(ns/3)):
            id_temp = id[:, j]
            vec[0:int(ns/3), 0] = np.transpose(id_temp) # Q matrix for the actual hand state
            vec[2*int(ns/3):3*int(ns/3), 0] = -np.transpose(id_temp)
            Q[i, :, :] = Q[i, :, :] + wprep[j, tprep_MPC]*vec @ np.transpose(vec)

    return R, Q


def basicKalman(Atot, Btot, Q, R, Htot, SigmaXi, SigmaOmega):
    # Calcuation of the optimal Kalman gains
    # Dimension parameters
    nStep = R.shape[0]
    na = Atot.shape[0]
    nc = Btot.shape[1]
    Sigma = SigmaXi
    ny = len(Htot)
    K = np.zeros(shape=(nStep,na,ny))
    # Recurrence
    for i in range(nStep):   
        K[i,:,:] = Atot @ Sigma@np.transpose(Htot)@np.linalg.inv(Htot @ Sigma @ np.transpose(Htot)+SigmaOmega)
        Sigma = SigmaXi + (Atot -K[i,:,:] @ Htot) @ Sigma @ np.transpose(Atot)
    return K



# Definition of the function basicLQG
def basicLQG(Atot, Btot, Q, R, SigmaXi):
  '''
  L,K = basicLQG(A,B,Q,R,H,SigmaXi,SigmaOmega)  
    Takes as input the matrices corresponding to the state space representation,
    the cost matrices, and the noie covariance matrices,

    Returns: time series of feedback gains and Kalman gains for simulations of 
    LQG control
  '''
  # Calculation of the optimal feedback gains
  # Dimension parameters
  nStep = R.shape[0]
  na = Atot.shape[0]
  nc = Btot.shape[1]
  #Recurrence
  S = np.zeros(shape=(nStep+1, na, na))
  S[nStep, :, :] = Q[nStep, :, :]
  L = np.zeros(shape=(nStep, nc, na))
  sadd = 0
  for i in reversed(range(nStep)):
    L[i, :, :] = np.linalg.inv(R[i, :, :] + np.transpose(Btot) @ S[i+1, :, :] @ Btot) @ np.transpose(Btot) @ S[i+1, :, :] @ Atot
    S[i, :, :] = Q[i, :, :] + np.transpose(Atot) @ S[i+1, :, :] @ (Atot - Btot @ L[i, :, :])
    sadd = sadd + np.trace(S[i+1, :, :] @ SigmaXi)
  return L

# mixed horizon control
def mixedLQG(Atot, Btot, Q, R, SigmaXi, nPrep):
  '''
  L,K = basicLQG(A,B,Q,R,H,SigmaXi,SigmaOmega)  
    Takes as input the matrices corresponding to the state space representation,
    the cost matrices, and the noie covariance matrices,

    Returns: time series of feedback gains and Kalman gains for simulations of 
    LQG control
  '''
  # Calculation of the optimal feedback gains. where preparatory period is solved using infinite horizon control
  # and the remaining is solved using finite horizon.
  # Calculation of the optimal feedback gains
  # Dimension parameters


  nStep = R.shape[0]
  na = Atot.shape[0]
  nc = Btot.shape[1]
  #Recurrence
  S = np.zeros(shape=(nStep+1, na, na))
  S[nStep, :, :] = Q[nStep, :, :]
  L = np.zeros(shape=(nStep, nc, na))
  sadd = 0
  for i in reversed(range(nStep)):
    L[i, :, :] = np.linalg.inv(R[i, :, :] + np.transpose(Btot) @ S[i+1, :, :] @ Btot) @ np.transpose(Btot) @ S[i+1, :, :] @ Atot
    S[i, :, :] = Q[i, :, :] + np.transpose(Atot) @ S[i+1, :, :] @ (Atot - Btot @ L[i, :, :])
    sadd = sadd + np.trace(S[i+1, :, :] @ SigmaXi)

  return L