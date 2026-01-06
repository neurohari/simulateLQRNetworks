import numpy as np
# import the state space control library so that a continuous SS can be solved for one time-step each time
from scipy.integrate import odeint

class limbNetwork:
    def __init__(self, params):
        self.m = params['m'] # mass 
        self.k = params['k'] # viscocity
        self.tau = params['tau'] # actuation time constant
        self.Win = params['Win'] # [Goal, Body] influence on network
        self.Wrec = params['Wrec'] # Network internal connectivity
        self.Wout = params['Wout'] # Network influence on the body
        self.dt = params['delta'] # time step
        self.nstates_n = self.Wrec.shape[0] # number of neurons in the network
        self.nstates_b = self.Win.shape[1] # number of states in the body

        self.neuron_tau_net = params['neuron_tau_net'] # neuron time constant

        self.A0 = np.zeros(shape=(8,8)) # state vector [x, xdot, fx, fxdot,    y, ydot, fy, fydot]
        self.B0 = np.zeros(shape=(8,2))
        self.A0[0, :] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        self.A0[1, :] = np.array([0, -self.k/self.m, 1/self.m, 1/self.m, 0, 0, 0, 0])
        self.A0[2, :] = np.array([0, 0, -1/self.tau, 0, 0, 0, 0, 0])
        self.A0[4, :] = np.array([0, 0, 0, 0, 0, 1, 0, 0])
        self.A0[5, :] = np.array([0, 0, 0, 0, 0, -self.k/self.m, 1/self.m, 1/self.m])
        self.A0[6, :] = np.array([0, 0, 0, 0, 0, 0, -1/self.tau, 0])

        self.B0[2, :] = np.array([1/self.tau, 0])
        self.B0[6, :] = np.array([0, 1/self.tau])

        # Initialize a local random number generator
        self.rng = np.random.RandomState(42)

        # append the network matrices with system matrices and perform disc2cont
        self.Atot0, self.Btot0 = self.appendNetworkMatrices()
        print(self.Atot0.shape, self.Btot0.shape)
        A_cont_temp = self.Atot0[0:109, 0:109] + 0.0
        B_cont_temp = self.Btot0[0:109, 0:100] + 0.0
        self.Atot, self.Btot = self.cont2disc()
        ns = self.Atot0.shape[0]
        nc = self.Btot0.shape[1]
        naug = 3 # augment the state space for home and target location states
        self.A_cont = np.zeros(shape=(naug*ns,naug*ns))
        self.A_cont[0:ns,0:ns] = A_cont_temp + 0.0
        self.B_cont = np.zeros(shape=(naug*ns,nc))
        self.B_cont[0:ns, 0:nc] = B_cont_temp + 0.0
        
        self.Htot = np.eye(3*(self.nstates_n + self.nstates_b + 1))
        self.states = np.zeros(((self.nstates_n + self.nstates_b + 1), 1)) # [ALLNEURONS, x, y, xdot, ydot, fx, fy, b]
        self.states[self.nstates_b:-1] = np.random.normal(5, 5, (self.nstates_n, 1)) # random offsets with 5Hz mean and 5Hz std
        self.estimate = np.zeros(((self.nstates_n + self.nstates_b + 1), 1))

        # determine the noise covariance properties of the body and measurement noise
        na = self.Atot.shape[0]
        self.SigmaXi = 0.5 * self.Btot @ (self.Btot.T)
        self.SigmaOmega = 0.5*np.max(self.SigmaXi)*np.eye(na)
        

        self.SigmaXi[3, 3] = 1*self.SigmaXi[9, 9]
        
        # assign motor noise to the force components
        self.SigmaXi[2, 2] = 1*self.SigmaXi[9, 9] + 0.0
        self.SigmaXi[6, 6] = 1*self.SigmaXi[9, 9] + 0.0
        self.SigmaXi[3, 3] = 1*self.SigmaXi[9, 9] + 0.0
        self.SigmaXi[7, 7] = 1*self.SigmaXi[9, 9] + 0.0


        return
    
    def appendNetworkMatrices(self):
        # Note that in Equation 15 in the article, we denoted state as [r, q, offset], but in this code we denote it as [q, r, offset]
        offset_body = np.zeros((self.nstates_b, 1))
        offset_neurons = 0 * self.rng.normal(0, 0.01, (self.nstates_n, 1)) #/ self.neuron_tau_net # random offsets also dividing by the neuron time constant
        temprow1 = np.concatenate((self.A0, self.Wout, offset_body), axis = 1)
        temprow2 = np.concatenate((self.Win, self.Wrec, offset_neurons), axis = 1)
        temprow3 = np.zeros((1, self.nstates_n+self.nstates_b+1))
        Atot0 = np.concatenate((temprow1, temprow2, temprow3), axis = 0)

        # all neurons can receive independent inputs
        Bnet = np.eye(self.nstates_n) #/ self.neuron_tau_net # this is to ensure neuron time constant is inlcuced in input term i.e., input term  = u / neuron_tau_net
        
        #### adding a row for the offset term ####
        Bnet = np.concatenate((Bnet, np.zeros((1, self.nstates_n))), axis = 0) # offset for each neuron

        ### No direct input to the body states, hence adding zero matrix on top of Bnet ###
        Btot0 = np.concatenate((np.zeros((self.nstates_b, self.nstates_n)), Bnet), axis = 0)

        return Atot0, Btot0
    
    def cont2disc(self):
        ''' to append target states, and to perform cont2disc conversion'''
        ns = self.Atot0.shape[0]
        nc = self.Btot0.shape[1]
        naug = 3 # to double the state number for a single target, if there were 2 seq. targets then the state would be *3
        #Augment matrices A and B with target state
        Atot = np.zeros(shape=(naug*ns,naug*ns))
        Atot[0:ns,0:ns] = self.Atot0 + 0.0
        Atot = np.eye(naug * ns) + (self.dt * Atot)
        
        Btot = np.zeros(shape=(naug*ns,nc))
        Btot[0:ns, 0:nc] = self.Btot0 + 0.0
        Btot = self.dt * Btot
        return Atot, Btot
    def disc2cont(self, a):
        na = a.shape[0]
        A = (a - np.eye(na))/self.dt
        return A
    
    def getSystemMatrices(self):
        return self.Atot, self.Btot, self.Htot
    
    @staticmethod
    def sysdyn(z, t, u, A, B):
        dzdt = A @ z 
        dzdt += B @ u
        return dzdt

    def nextState(self, Tor): # Tor is supposed to be torques (xtorque; ytorque)
        # Here we take Torques as inputs and compute the state update using (x_dot = A*x + B*Tor)
        xi = np.random.multivariate_normal(np.zeros(self.Atot.shape[0]), self.SigmaXi, tol=1e-8)
        cur_state = self.states + 0.0

        # updating next state with the naive method - CAUTION when using with network dynamics! 
        #next_state = (self.Atot @ cur_state) + (self.Btot @ Tor) + xi.T
        # updating the next state using better continuous differential equ. solvers (ss, odeint etc.,)
        next_state = odeint(self.sysdyn, cur_state, np.linspace(0,0.01,2), args=(Tor, self.A_cont, self.B_cont))[-1] + 0*xi.T


        # update the state of the point-mass
        self.states = next_state + 0.0
        return next_state
    
    def nextEstimate(self, gains, Tor):
        cur_estimate = self.estimate + 0.0

        omega = np.random.multivariate_normal(
            np.zeros(self.Atot.shape[0]), self.SigmaOmega, tol=1e-8)
        y = (self.Htot @ self.states) + omega
        innovation = y - (self.Htot @ cur_estimate)

        # updating next estimate with the naive method - CAUTION when using with network dynamics! 
        #next_estimate = (self.Atot @ cur_estimate) + (self.Btot @ Tor) + (gains @ innovation)
        # updating the next estimate using better continuous differential equ. solvers (ss, odeint etc.,)
        next_estimate = odeint(self.sysdyn, cur_estimate, np.linspace(0,0.01,2), args=(Tor, self.A_cont, self.B_cont))[-1] + (gains @ innovation)


        # update the kalman estimate of the point-mass
        self.estimate = next_estimate + 0.0
        return next_estimate, innovation
    
    def reset(self, init_state):
        # [ALLNEURONS, x, y, xdot, ydot, fx, fy]
        self.states = init_state + 0.0
        self.estimate = init_state + 0.0
        return
    






    ###### MODIFIED DYNAMICS CLASS FOR FORCE-FIELD TASK and INTERTIAL LOAD TASK ########
    ################################## class of another same mass but this time with a force field #################
class limbNetwork_modifieddynamics:
    def __init__(self, params):
        self.m = params['m'] # mass 
        self.k = params['k'] # viscocity
        self.tau = params['tau'] # actuation time constant
        self.Win = params['Win'] # [Goal, Body] influence on network
        self.Wrec = params['Wrec'] # Network internal connectivity
        self.Wout = params['Wout'] # Network influence on the body
        self.dt = params['delta'] # time step
        self.nstates_n = self.Wrec.shape[0] # number of neurons in the network
        self.nstates_b = self.Win.shape[1] # number of states in the body
        # set the force field parameters
        self.forcefield_param = params['forcefield_param'] # force field strength
        self.inertial_param = params['inertial_param'] # inertial load strength

        # scale the inertial load in a interital task
        self.m = params['m'] * self.inertial_param

        self.neuron_tau_net = params['neuron_tau_net'] # neuron time constant

        self.A0 = np.zeros(shape=(8,8)) # state vector [x, xdot, fx, fxdot,    y, ydot, fy, fydot]
        self.B0 = np.zeros(shape=(8,2))
        self.A0[0, :] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        self.A0[1, :] = np.array([0, -self.k/self.m, 1/self.m, 0, 0, self.forcefield_param/self.m, 0, 0])
        self.A0[2, :] = np.array([0, 0, -1/self.tau, 0, 0, 0, 0, 0])
        self.A0[4, :] = np.array([0, 0, 0, 0, 0, 1, 0, 0])
        self.A0[5, :] = np.array([0, self.forcefield_param/self.m, 0, 0, 0, -self.k/self.m, 1/self.m, 0])
        self.A0[6, :] = np.array([0, 0, 0, 0, 0, 0, -1/self.tau, 0])

        self.B0[2, :] = np.array([1/self.tau, 0])
        self.B0[6, :] = np.array([0, 1/self.tau])

        # Initialize a local random number generator
        self.rng = np.random.RandomState(42)
        
        # append the network matrices with system matrices and perform disc2cont
        self.Atot0, self.Btot0 = self.appendNetworkMatrices()
        print(self.Atot0.shape, self.Btot0.shape)
        A_cont_temp = self.Atot0[0:109, 0:109] + 0.0
        B_cont_temp = self.Btot0[0:109, 0:100] + 0.0
        self.Atot, self.Btot = self.cont2disc()
        ns = self.Atot0.shape[0]
        nc = self.Btot0.shape[1]
        naug = 3 # augment the state space for home and target location states
        self.A_cont = np.zeros(shape=(naug*ns,naug*ns))
        self.A_cont[0:ns,0:ns] = A_cont_temp + 0.0
        self.B_cont = np.zeros(shape=(naug*ns,nc))
        self.B_cont[0:ns, 0:nc] = B_cont_temp + 0.0
        
        self.Htot = np.eye(3*(self.nstates_n + self.nstates_b + 1))
        self.states = np.zeros(((self.nstates_n + self.nstates_b + 1), 1)) # [x,xdot,fx,fydot,    y,ydot,fy,fydot,   ALLNEURONS, offset]
        self.states[self.nstates_b:-1] = np.random.normal(5, 5, (self.nstates_n, 1)) # random offsets with 5Hz mean and 5Hz std
        self.estimate = np.zeros(((self.nstates_n + self.nstates_b + 1), 1))

        # determine the noise covariance properties of the body and measurement noise
        na = self.Atot.shape[0]
        self.SigmaXi = 0.5 * self.Btot @ (self.Btot.T)
        self.SigmaOmega = 0.5*np.max(self.SigmaXi)*np.eye(na)
        

        self.SigmaXi[3, 3] = 1*self.SigmaXi[9, 9]
        
        # assign motor noise to the force components
        self.SigmaXi[2, 2] = 1*self.SigmaXi[9, 9] + 0.0
        self.SigmaXi[6, 6] = 1*self.SigmaXi[9, 9] + 0.0
        self.SigmaXi[3, 3] = 1*self.SigmaXi[9, 9] + 0.0
        self.SigmaXi[7, 7] = 1*self.SigmaXi[9, 9] + 0.0
        
        


        return
    
    def appendNetworkMatrices(self):
        offset_body = np.zeros((self.nstates_b, 1))
        offset_neurons = 0 * self.rng.normal(0, 0.01, (self.nstates_n, 1)) #/ self.neuron_tau_net # random offsets while also accounting for neuron time constants
        temprow1 = np.concatenate((self.A0, self.Wout, offset_body), axis = 1)
        temprow2 = np.concatenate((self.Win, self.Wrec, offset_neurons), axis = 1)
        temprow3 = np.zeros((1, self.nstates_n+self.nstates_b+1))
        Atot0 = np.concatenate((temprow1, temprow2, temprow3), axis = 0)

        # all neurons can receive independent inputs
        Bnet = np.eye(self.nstates_n) #/ self.neuron_tau_net # this ensures that the neural time constant tau is considered in th input B matrix

        #### adding a row for the offset term ####
        Bnet = np.concatenate((Bnet, np.zeros((1, self.nstates_n))), axis = 0) # offset for each neuron

        ### No direct input to the body states, hence adding zero matrix on top of Bnet ###
        Btot0 = np.concatenate((np.zeros((self.nstates_b, self.nstates_n)), Bnet), axis = 0)

        return Atot0, Btot0
    
    def cont2disc(self):
        ''' to append target states, and to perform cont2disc conversion'''
        ns = self.Atot0.shape[0]
        nc = self.Btot0.shape[1]
        naug = 3 # to double the state number for a single target, if there were 2 seq. targets then the state would be *3
        #Augment matrices A and B with target state
        Atot = np.zeros(shape=(naug*ns,naug*ns))
        Atot[0:ns,0:ns] = self.Atot0 + 0.0
        Atot = np.eye(naug * ns) + (self.dt * Atot)
        
        Btot = np.zeros(shape=(naug*ns,nc))
        Btot[0:ns, 0:nc] = self.Btot0 + 0.0
        Btot = self.dt * Btot
        return Atot, Btot
    def disc2cont(self, a):
        na = a.shape[0]
        A = (a - np.eye(na))/self.dt
        return A
    
    def getSystemMatrices(self):
        return self.Atot, self.Btot, self.Htot
    
    @staticmethod
    def sysdyn(z, t, u, A, B):
        dzdt = A @ z 
        dzdt += B @ u
        return dzdt

    def nextState(self, Tor): # Tor is supposed to be torques (xtorque; ytorque)
        # Here we take Torques as inputs and compute the state update using (x_dot = A*x + B*Tor)
        xi = np.random.multivariate_normal(np.zeros(self.Atot.shape[0]), self.SigmaXi, tol=1e-8)
        cur_state = self.states + 0.0

        # updating next state with the naive method - CAUTION when using with network dynamics! 
        #next_state = (self.Atot @ cur_state) + (self.Btot @ Tor) + xi.T
        # updating the next state using better continuous differential equ. solvers (ss, odeint etc.,)
        next_state = odeint(self.sysdyn, cur_state, np.linspace(0,0.01,10), args=(Tor, self.A_cont, self.B_cont))[-1] + 0*xi.T


        # update the state of the point-mass
        self.states = next_state + 0.0
        return next_state
    
    def nextEstimate(self, gains, Tor):
        cur_estimate = self.estimate + 0.0

        omega = np.random.multivariate_normal(
            np.zeros(self.Atot.shape[0]), self.SigmaOmega, tol=1e-8)
        y = (self.Htot @ self.states) + omega
        innovation = y - (self.Htot @ cur_estimate)

        # updating next estimate with the naive method - CAUTION when using with network dynamics! 
        #next_estimate = (self.Atot @ cur_estimate) + (self.Btot @ Tor) + (gains @ innovation)
        # updating the next estimate using better continuous differential equ. solvers (ss, odeint etc.,)
        next_estimate = odeint(self.sysdyn, cur_estimate, np.linspace(0,0.01,10), args=(Tor, self.A_cont, self.B_cont))[-1] + (gains @ innovation)


        # update the kalman estimate of the point-mass
        self.estimate = next_estimate + 0.0
        return next_estimate, innovation
    
    def reset(self, init_state):
        # [ALLNEURONS, x, y, xdot, ydot, fx, fy]
        self.states = init_state + 0.0
        self.estimate = init_state + 0.0
        return



