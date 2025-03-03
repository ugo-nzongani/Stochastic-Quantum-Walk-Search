# Stochastic-Quantum-Walk-Search

This repo contains the code used to generate the datas for the paper "Noisy-enhanced quantum search on complex networks".

The Stochastic Quantum Walk Search (SQWS) is a continuous-time evolution used to solve a searching problem on graph by mixing unitary and non-unitary dynamics. The walker has to reach a target vertex on the graph. An additional trapping vertex called the "sink" is connected with an irreversible transition to the target vertex of the search. The SQWS uses two tunable parameters: $\gamma>0$ and $\omega\in [0,1]$. The first controls the strength of the quantum walk exploration in the Hamiltonian, and the second controls the interplay between unitary and non-unitary dynamics.
