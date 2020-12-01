import time
import sys
import os
# from graph import graph # my own code for graph object
import heapq
from collections import deque,defaultdict
import argparse
import random
from graph import graph
from approx import mdg

# read data and construct a graph object
def parse_edges(filename):
    # parse edges from graph file to create your graph object
    # filename: string of the filename
    f = open(filename, "r")
    n_vertices, n_edges, _ = f.readline().split(' ')
    n_vertices, n_edges = int(n_vertices), int(n_edges)

    G = graph()  # create a graph

    # add edges to the graph
    for i in range(1, n_vertices + 1):
        neighbors = f.readline().rstrip().split(' ')
        for neighbor in neighbors:
            if neighbor != '':
                if int(neighbor) > i:
                    G.add_edge(i, int(neighbor))

    return G 

def approx2(G,seed):
    # G: a graph object
    # return: the number of vertices of a 2-approximation solution to the min vertex cover problem of G
    
    S = set()

    shuffled_vertices = list(G.vert_dict.values())
    
    # random.Random(seed).shuffle(shuffled_vertices)

    for v in shuffled_vertices:
        if v in S:
            continue

        for neighbor in v.get_connections():        
            if v.id not in S and neighbor.id not in S:
                S.add(v.id)
                S.add(neighbor.id)

    return S, len(S) # dividing the result by 2 gives us a lower bound of the minimum vertex cover of G


def Gprime(G, Cprime):
    G_prime = G.copy()
    for v in Cprime:
        G_prime = G_prime.remove_vertex(v,inplace = True)
    return G_prime

def find_next(G_prime, Explored):
    V_prime = G_prime.get_vertices()
    for v in sorted(V_prime, key = lambda x: - G_prime.get_vertex_degree(x)):
        if v not in Explored:
            return v


class node:
    # decision node in our Decision Tree
    def __init__(self,id, parent,lb,state):

        self.id = id # the id of the vertex in the original graph G
        self.parent = parent # parent of the current decision node
        self.lb = lb # lower bound of the current decision node 
        self.state = state # state == 1 if we consider including vertex(id) in the vertex cover
    
    def __lt__(self, other): 
        return self.lb < other.lb
         

def Branch_and_Bound(G, start_time, cutoff, fo, upperBound, seed):
    """
    Find min vertex using a branch and bound algorithm.

    Consider all possible vertex covers and prune nonpromising ones using lower bounds and upper bounds

    Input: a networkx undirected graph
    Returns: the minimum vertex cover
    """  

    opt_cover, opt_num = approx2(G, seed)
    fo.write(str(time.time() - start_time) + ',' + str(opt_num) + "\n")    

    # initial lowerbound
    LB = opt_num // 2   

    # initial vertex to consider
    first_vertex = find_next(G, set())


    # number of uncovered edges
    num_uncov_edges0 = G.num_edges # not selecting the vertex[vertices_order[0]]
    num_uncov_edges1 = G.num_edges - G.get_vertex_degree(first_vertex) # selecting the vertex[vertices_order[0]]

    # initialize priority queue with the first two decision nodes and their corresponding number of uncovered edges
    pqueue = [(num_uncov_edges0, node(first_vertex,None,LB,0)),(num_uncov_edges1,node(first_vertex,None,LB,1))]
    heapq.heapify(pqueue)

    new_node0, new_node1 = None, None


    while pqueue and ((time.time() - start_time) < cutoff) and opt_num > upperBound:

        # get the most promising decision node with the least number of uncovered edges, then the highest lower bound
        num_uncov_edges,Dnode = heapq.heappop(pqueue)

        parent = Dnode.parent
        node_id = Dnode.id  


        if Dnode == new_node1:
            Cprime.add(new_node_id)
            G_prime = G_prime1
        elif Dnode == new_node0:
            explored.add(new_node_id)
        else:           
            # initialzie cover set Cprime and explored vertices     
            Cprime = set([Dnode.id]) if Dnode.state else set() # vertices in the conver set
            explored = set() if Dnode.state else set([Dnode.id]) # vertices explored but not in the cover set

            # trace back the decision tree to find Cprime and explored
            while parent:
        
                if parent.state:
                    Cprime.add(parent.id)
                else:
                    explored.add(parent.id)

                parent = parent.parent
        
            G_prime = Gprime(G, Cprime)
        
        # find the vertex with the highest degree in G_prime that has not been explored
        new_node_id = find_next(G_prime, explored)
        

        if not new_node_id:
            continue # prune it

        
        # Branch 1: add vertex(new_node_id) to the cover set

        cover_size = len(Cprime) + 1  
        new_cover = Cprime.union(set([new_node_id]))
    
        G_prime1 = G_prime.remove_vertex(new_node_id,inplace = True)
   
        if not G_prime1.get_vertices():
            # solution found
            if cover_size < opt_num:
                # update solution
                opt_num = cover_size
                opt_cover = new_cover 
                fo.write(str(time.time() - start_time) + ',' + str(opt_num) + "\n")
                # print('Optimal:', opt_num)
            continue

        
        LowerBound = cover_size + approx2(G_prime1, seed)[1]//2

        if LowerBound < opt_num:
            new_node1 = node(new_node_id, Dnode, LowerBound,1)
            heapq.heappush(pqueue, (G_prime.num_edges, new_node1))
    

        # Branch 2: don't add vertex(new_node_id) to the cover set
        # check deadend
        deadend = False
        for new_node_neigh in G.get_vertex(new_node_id).get_connections():
            if new_node_neigh.id in explored:
                deadend = True
                break

        if not deadend and Dnode.lb < opt_num:
            new_node0 = node(new_node_id, Dnode, Dnode.lb,0)
            heapq.heappush(pqueue, (num_uncov_edges, new_node0))

    return opt_num, opt_cover

opt_cutoff = {'karate':14, 'football':94, 'jazz':158, 'email':594, 'delaunay_n10':703,'netscience':899, 'power':2203,'as-22july06':3303,'hep-th':3926,'star2':4542,'star':6902}

def main(graph, algo, cutoff, seed):
    G = parse_edges(graph)
    graph_name = graph.split('/')[-1].split('.')[0]

    if graph_name not in opt_cutoff:
        return

    sol_file = "_".join([graph_name, algo, str(cutoff)]) + '.sol'
    trace_file = "_".join([graph_name, algo, str(cutoff)]) + '.trace'
    output_dir = './BnB_output/'

    start_time = time.time()

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fo = open(os.path.join(output_dir, trace_file), 'w')

    if algo == 'BnB':
        num_vc_nodes, vc = Branch_and_Bound(G, start_time, cutoff, fo, opt_cutoff[graph_name], seed)
        fo.close()


    total_time = round((time.time() - start_time), 5)
    print('BnB Algo Runtime: ' + str(total_time))

    

    with open(os.path.join(output_dir, sol_file), 'w') as f:
        f.write(str(num_vc_nodes) + "\n")
        f.write(','.join([str(n) for n in sorted(vc)]))
    f.close()


# Run as executable from terminal
if __name__ == '__main__':
    #parse arguments in the following format: 
    # python Python/BnB.py -inst DATA/jazz.graph -alg BnB -time 600 -seed 30
    parser = argparse.ArgumentParser(description='Run algorithm with specified parameters')
    parser.add_argument('-inst', type=str, required=True, help='graph file')
    parser.add_argument('-alg', type=str, required=True, help='algorithm to use')
    parser.add_argument('-time', type=float, default=600, required=False, help='runtime cutoff for algorithm')
    parser.add_argument('-seed', type=int, default=30, required=False, help='random seed for algorithm')
    args = parser.parse_args()

    main(args.inst, args.alg, args.time, args.seed)

        