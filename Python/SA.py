#!/usr/bin/python
# CSE6140 proj
# This is an example of how your experiments should look like.
# Feel free to use and modify the code below, or write your own experimental code, as long as it produces the desired output.
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
        temp_G = list(G.nodes())
        VC = sorted(list(zip(list(dict(G.degree(temp_G)).values()), temp_G)), reverse=False)
        i = 0
        while(i < len(VC) and (time.time() - start_time) < cutoff):
            flag=True
            for x in G.neighbors(VC[i][1]):
                if x not in temp_G:
                    flag = False
            if flag:    
                temp_G.remove(VC[i][1])            
            i=i+1
        fo.write(str(time.time()-start_time) + "," + str(len(temp_G)) + "\n")
        print('Initial Solution:({}) {}'.format(len(temp_G), temp_G))
        return temp_G

    def simulate_annealing(self, G, fo, sol, cutoff, nV, start_time, input_file, upperBound):
        temp = 0.15     
        update_sol = sol.copy()
        uncov_old = []
        num_eges= G.number_of_edges()
        while ((time.time() - start_time) < cutoff):
            temp = 0.95 * temp 
            while not uncov_old:
                update_sol = sol.copy()
                fo.write(str(time.time()-start_time) + "," + str(len(update_sol)) + "\n")
                delete = random.choice(sol)
                for x in G.neighbors(delete):
                    if x not in sol:
                        uncov_old.append(x)
                        uncov_old.append(delete)
                sol.remove(delete)     

            # del node
            current = sol.copy()
            uncov_new = uncov_old.copy()
            delete = random.choice(sol)
            for x in G.neighbors(delete):
                if x not in sol:
                    uncov_old.append(x)
                    uncov_old.append(delete)            
            sol.remove(delete)   

            # add node
            enter = random.choice(uncov_old)
            sol.append(enter)
            for x in G.neighbors(enter):
                if x not in sol:
                    uncov_old.remove(enter)
                    uncov_old.remove(x)

            cost_new = len(uncov_new)
            cost_old = len(uncov_old)
            if cost_new < cost_old: 
                prob = math.exp(float(cost_new - cost_old)/float(temp))
                num = round(random.uniform(0,1),10)
                if num > prob:    
                    sol = current.copy()
                    uncov_old = uncov_new.copy()

        return sorted(update_sol)

    def main(self, input_file, cutoff, seed):
        random.seed(seed)

        fo = open('DATA/Traces/{}_{}_{}.trace'.format(input_file.split('/')[1].split('.')[0], cutoff, seed), 'w')
        G, nV, nE = self.parse_edges(input_file)

        start_time = time.time()
        G_init = G.copy()
        sol = self.initial_solution(G=G_init, fo=fo, start_time=start_time, cutoff=cutoff, input_file=input_file)
        final_solution = self.simulate_annealing(G, fo, sol, cutoff, nV, start_time, input_file, opt_cutoff.get(input_file, 10))
        fo.close()
        print('Final Solution: ({}) {}'.format(len(final_solution), final_solution))

        fo = open('DATA/Solutions/{}_{}_{}.sol'.format(input_file.split('/')[1].split('.')[0], cutoff, seed), 'w')
        fo.write('{}\n'.format(len(final_solution)))
        fo.write(','.join([str(v) for v in final_solution]))
        fo.close()

        total_time = (time.time() - start_time)
        print('SA Runtime (s): {}'.format(total_time))


if __name__ == '__main__':
    # python temp.py -input DATA/dummy1.graph -time 600 -seed 1000
    parser=argparse.ArgumentParser(description='Input parser for SA')
    parser.add_argument('-input',action='store',type=str,required=True,help='Input graph datafile')
    parser.add_argument('-time',action='store',default=600,type=float,required=False,help='Cutoff running time for algorithm')
    parser.add_argument('-seed',action='store',default=1000,type=int,required=False,help='Random Seed for algorithm')       
    args=parser.parse_args()

    # run the experiments
    runexp = RunExperiments()
    runexp.main(args.input, args.time, args.seed)
