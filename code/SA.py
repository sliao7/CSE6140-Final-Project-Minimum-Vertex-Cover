#!/usr/bin/python
# Simulated Annealing (SA) is used to approximate the global optimal of the target function, 
# it is proposed based on the idea of the physical process of annealing.
import time
import sys

import math
import random
import argparse
import numpy as np
import networkx as nx

class SA:

    def parse_edges(self, filename):
        G = nx.Graph()
        lines = open(filename, 'r').readlines()
        nV, nE, _ = lines[0].split()
        for i in range(1, int(nV)+1):
            vs = lines[i].split()
            for v in vs:
                G.add_edge(i, int(v))
        return G, int(nV), int(nE)

    def initial_solution(self, G, fo, start_time, cutoff, input_file):
        # create initial solution via removing nodes with more connection (lower bound)

        _G = list(G.nodes())
        VC = sorted(list(zip(list(dict(G.degree(_G)).values()), _G)), reverse=False)
        i = 0
        while (i < len(VC) and (time.time() - start_time) < cutoff):
            check = True
            for x in G.neighbors(VC[i][1]):
                if x not in _G:
                    check = False
            if check:    
                _G.remove(VC[i][1])            
            i += 1
        fo.write(str(time.time()-start_time) + "," + str(len(_G)) + "\n")
        print('Initial Solution:({}) {}'.format(len(_G), _G))
        return _G

    def simulate_annealing(self, G, fo, S, cutoff, nV, start_time, input_file, upperBound):
        T = 0.8   
        S_ret = S.copy()
        S_best = []
        while ((time.time() - start_time) < cutoff):
            T = 0.95 * T 

            # looking for a better solution with less vertice
            while not S_best:
                S_ret = S.copy()
                fo.write(str(time.time()-start_time) + "," + str(len(S_ret)) + "\n")
                delete_v = random.choice(S)
                for v in G.neighbors(delete_v):
                    if v not in S:
                        S_best.append(v)
                        S_best.append(delete_v)
                S.remove(delete_v)     


            # del node

            S_current = S.copy()
            uncovered_S = S_best.copy()
            delete_v = random.choice(S)
            for v in G.neighbors(delete_v):
                if v not in S:
                    S_best.append(v)
                    S_best.append(delete_v)            
            S.remove(delete_v)   


            # add node

            add_v = random.choice(S_best)
            S.append(add_v)
            for v in G.neighbors(add_v):
                if v not in S:
                    S_best.remove(v)
                    S_best.remove(add_v)

            # accept a new solution based on the probability which is proportional to the 
            # difference between the quality of the best solution and the current solution, and the temperature. 
            if len(uncovered_S) < len(S_best): 
                p = math.exp(float(len(uncovered_S) - len(S_best))/T)
                alpha = random.uniform(0, 1)
                if alpha > p:    
                    S = S_current.copy()
                    S_best = uncovered_S.copy()

        return S_ret

    def main(self, input_file, cutoff, seed):
        random.seed(seed)

        fo = open('DATA/Traces/{}_{}_{}.trace'.format(input_file.split('/')[1].split('.')[0], cutoff, seed), 'w')
        G, nV, nE = self.parse_edges(input_file)

        start_time = time.time()
        G_init = G.copy()
        S_init = self.initial_solution(G=G_init, fo=fo, start_time=start_time, cutoff=cutoff, input_file=input_file)
        final_solution = self.simulate_annealing(G, fo, S_init, cutoff, nV, start_time, input_file, opt_cutoff.get(input_file, 10))
        fo.close()
        print('Final Solution: ({}) {}'.format(len(final_solution), final_solution))

        fo = open('DATA/Solutions/{}_{}_{}.sol'.format(input_file.split('/')[1].split('.')[0], cutoff, seed), 'w')
        fo.write('{}\n'.format(len(final_solution)))
        fo.write(','.join([str(v) for v in final_solution]))
        fo.close()

        total_time = (time.time() - start_time)
        print('SA Runtime (s): {}'.format(total_time))


if __name__ == '__main__':
    # python SA.py -input DATA/dummy1.graph -time 600 -seed 1000
    # python Python/main.py -inst DATA/dummy2.graph -alg SA -time 600 -seed 600
    parser=argparse.ArgumentParser(description='Input parser for SA')
    parser.add_argument('-input',action='store',type=str,required=True,help='Input graph datafile')
    parser.add_argument('-time',action='store',default=600,type=float,required=False,help='Cutoff running time for algorithm')
    parser.add_argument('-seed',action='store',default=1000,type=int,required=False,help='Random Seed for algorithm')       
    args=parser.parse_args()

    # run the experiments
    runexp = RunExperiments()
    runexp.main(args.input, args.time, args.seed)
