import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qutip import *
import math
from scipy.optimize import Bounds, minimize, BFGS, dual_annealing

class QSWSearch:
    
    def __init__(self,graph,marked_node,w=0.,initial_state=None,sink_rate=0.,gamma=None,dt=1.,time=None,save=False,caruso=False):
        self.graph = graph
        self.adjacency = nx.adjacency_matrix(graph).toarray()
        self.laplacian = nx.laplacian_matrix(graph).toarray() # D-A
        self.marked_node = marked_node
        self.lambda_max = 0
        self.laplacian = self.normalize_laplacian()
        self.eigenvalues, self.eigenvectors = self.eigen()
        self.spectral_gap = 1-self.eigenvalues[-2]
        self.coefficients = self.get_coefficients()
        
        self.n_nodes = len(self.adjacency)
        if gamma == None:
            self.gamma = self.gamma_init_optimal(nx.laplacian_matrix(graph).toarray())
        else:
            self.gamma = gamma
        self.success = []
        self.w = w
        self.hamiltonian = self.gamma*self.laplacian + self.marked_projector()
        self.lindblad_op = self.laplacian_matrix_sink()
        self.state = []
        self.dt = dt
        if time == None:
            self.time = 3*np.sqrt(self.n_nodes)
        else:
            self.time = time
        self.time_list = np.arange(0,time+self.dt,self.dt)
        self.sink_rate = sink_rate
        self.save = save
        self.caruso = caruso
        if initial_state == None:
            self.initial_state = ket2dm(sum(basis(self.n_nodes+1, i) for i in range(self.n_nodes)) * (1/np.sqrt(self.n_nodes)))
        else:
            self.initial_state = initial_state
        self.collapse = self.collapse_operators()
                       
    ## Code for analysis: https://arxiv.org/pdf/2004.12686
    def eigen(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.laplacian)
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = []
        sorted_eigenvectors = []
        for i in range(len(eigenvalues)):
            sorted_eigenvalues.append(eigenvalues[sorted_indices[i]])
            sorted_eigenvectors.append(eigenvectors[:,sorted_indices[i]])
        return sorted_eigenvalues, sorted_eigenvectors

    def sk(self,k):
        s = 0
        for i in range(len(self.eigenvalues)-1):
            s += np.abs(self.coefficients[i])**2/(1-self.eigenvalues[i])**k
        return s

    def get_coefficients(self):
        oracle = np.zeros(len(self.eigenvectors))
        oracle[self.marked_node] = 1.
        return np.linalg.solve(self.eigenvectors, oracle)

    def normalize_laplacian(self):
        eigenvalues, _ = np.linalg.eig(self.laplacian)
        lambda_max = np.max(eigenvalues)
        #print('ugo:',lambda_max)
        return np.identity(self.laplacian.shape[0])-self.laplacian/lambda_max
    
    def gamma_init_optimal(self,laplacian):
        eigenvalues, _ = np.linalg.eig(laplacian)
        lambda_max = np.max(eigenvalues)
        self.lambda_max = lambda_max
        s1 = self.sk(1)
        #print(s1,np.linalg.norm(laplacian),lambda_max)
        #return s1/lambda_max
        return np.real(s1/lambda_max)
    
    def search_time(self):
        s1 = self.sk(1)/self.lambda_max
        s2 = self.sk(2)/self.lambda_max
        eps = 1/self.n_nodes # if graph is regular
        asympt_cst = 10
        return asympt_cst * np.real(np.sqrt(s2)/(np.sqrt(eps)*s1))
    
    def ugo(self):
        new_laplacian = self.laplacian.copy()
        for i in range(len(new_laplacian)):
            if i != self.marked_node:
                new_laplacian[i,self.marked_node] = 0.
            else:
                new_laplacian[i,self.marked_node] = 1.
        return new_laplacian
    
    def random_walk(self):
        deg = np.sum(self.adjacency, axis=0)
        adj = -self.adjacency_matrix_sink()
        for i in range(len(adj)):
            adj[i,i] += deg[i]
        eigenvalues, _ = np.linalg.eig(adj)
        lmax = max(eigenvalues)
        return -adj/lmax
    
    ##
    
    def adjacency_matrix_sink(self):
        adj = self.adjacency.copy()
        for i in range(len(adj)):
            if i != self.marked_node:
                adj[i,self.marked_node] = 0.
            else:
                adj[i,self.marked_node] = 1.
        return np.array(adj)
    
    def laplacian_matrix_sink(self):
        deg = np.sum(self.adjacency, axis=0)
        adj = self.adjacency_matrix_sink()
        for i in range(len(adj)):
            adj[i,i] = -deg[i]
        return adj/np.linalg.norm(adj)
    
    def marked_projector(self):
        proj = np.zeros((self.n_nodes,self.n_nodes))
        proj[self.marked_node,self.marked_node] = 1.
        return proj
        
    def plot_graph(self):
        nx.draw(self.graph, with_labels=True, node_color='aqua', edge_color='black', node_size=40, font_size=13)
        plt.show()
        
    def success_probability(self):
        success = []
        if self.sink_rate == 0.:
            node = self.marked_node
        else:
            node = -1
        if self.save:
            for i in range(len(self.state.states)):
                success.append(np.real(self.state.states[i].data.diagonal()[node]))
        else:
            success.append(np.real(self.state.final_state.data.diagonal()[node]))
        return success
    
    def collapse_operators(self):
        N = self.n_nodes
        if self.sink_rate == 0.:
            T = self.random_walk()
        else:
            #T = self.laplacian - np.identity(self.laplacian.shape[0]) # A-D, put a minus sign if using the not-normalized laplacian, we sub the identity to only have T = I-L/lambda_max-I=-L/lambda_max
            T = self.random_walk()
        S = np.zeros([N + 1, N + 1])  # transition matrix to the sink
        S[N, self.marked_node] = np.sqrt(self.sink_rate)
        if math.isclose(self.w, 0):
                L = []
        else:
            L = [np.sqrt(self.w * T[i, j]) * (basis(N + 1, i) * basis(N + 1, j).dag())
                 for i in range(N) for j in range(N) if T[i, j] > 0]  # set of Lindblad operators
        L.append(Qobj(S))  # add the sink transfer
        return L
    
    def search(self,x):
        self.gamma = x[0]
        N = self.n_nodes
        self.hamiltonian = self.gamma*self.laplacian + self.marked_projector()
            
        if math.isclose(self.w, 1):
            H = Qobj(0 * np.pad(self.hamiltonian, [(0, 1), (0, 1)], 'constant'))
            # Qobj(np.zeros((N+1, N+1))) seems slower
        else:
            H = Qobj((1 - self.w) * np.pad(self.hamiltonian, [(0, 1), (0, 1)], 'constant'))
            # add zero padding to account for the sink and multiply for (1-w)
       
        nsteps = 3000#1000
        opts = Options(store_states=self.save, store_final_state=True, nsteps=nsteps)
        result = mesolve(H, self.initial_state, self.time_list, self.collapse, options=opts)  # solve master equation

        self.state = result
        self.success = self.success_probability()
        
    def plot_search(self):
        if self.save:
            plt.plot(self.time_list,self.success,color='blue')
            plt.xlabel('Time')
            plt.ylabel('Probability')
            plt.show()
        else:
            print('------QSWSearch------')
            print('Success probability: ',self.success[-1])
            print('Time: ',self.time)
            
    def attributes(self):
        print('--------QSWSearch--------')
        print('w =',self.w)
        print('gamma =',self.gamma)
        print('marked node =',self.marked_node)
        print('sink_rate =',self.sink_rate)
        print('initial_state =',self.initial_state)
        
    def cost_function(self,x):
        self.search(x)
        
        a = self.time_list[0]
        b = self.time_list[-1]
        dt = self.dt
        x_list = [self.time_list[j] for j in range(len(self.time_list))]
        y = self.success
        if self.sink_rate == 0:
            return -max(y)
        else:
            return -np.trapz(y,x_list,dt)/self.time_list[-1]


