import time
import sys
import os
# from graph import graph # my own code for graph object
import heapq
from collections import deque,defaultdict
import argparse
from graph import graph

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

def approx2(G):
    # G: a graph object
    # return: the number of vertices of a 2-approximation solution to the min vertex cover problem of G

    S = set()
    for v in G:
        if v in S:
            continue
        for neighbor in v.get_connections():
            
            if neighbor.id < v.id:
                continue
            if v.id not in S and neighbor.id not in S:
                S.add(v.id)
                S.add(neighbor.id)
    return len(S) # dividing the result by 2 gives us a lower bound of the minimum vertex cover of G


def Find_Vprime(G, Cprime):
    'Cprime: a list of numbers in {1, 2, ..., n}'
    'Return: a set of vertices Vprime of the graph Gprime'
    Vprime = set()
#     Cprime = set(Cprime)
    
    for node in G.get_vertices():
        if node in Cprime:
            continue
            
        for neighbor in G.get_vertex(node).adjacent:
            if neighbor.id not in Cprime:
                Vprime.add(node)
                break
    return Vprime


def Gprime(G,Vprime):
    'Construct graph Gprime from Vprime'
    G_prime = graph()
    for node in Vprime:
        for neighbor in G.get_vertex(node).adjacent:
            if neighbor.id > node and neighbor.id in Vprime:
                G_prime.add_edge(node,neighbor.id)
    return G_prime

def find_next(V_prime, G_prime, Explored):
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
        return self.lb > other.lb
         

def Branch_and_Bound(G, start_time, cutoff):
    """
    Find min vertex using a branch and bound algorithm.

    Consider all possible vertex covers and prune nonpromising ones using lower bounds and upper bounds

    Input: a networkx undirected graph
    Returns: the minimum vertex cover
    """
    n = G.num_vertices   
    vertices = G.get_vertices()
    # sort vertices in degree 
    vertices.sort(key = lambda x: -len(G.get_vertex(x).adjacent)) 

    opt_cover = vertices
    opt_num = n

    # initial lowerbound
    LB = approx2(G) / 2   

    # to get the order of vertex using vertex id
    vertices_order = {vertex: i for i, vertex in enumerate(vertices)}

    # number of uncovered edges
    num_uncov_edges0 = G.num_edges # not selecting the vertex[vertices_order[0]]
    num_uncov_edges1 = G.num_edges - G.get_vertex_degree(vertices[0]) # selecting the vertex[vertices_order[0]]

    # initialize priority queue with the first two decision nodes and their corresponding number of uncovered edges
    pqueue = [(num_uncov_edges0, node(vertices[0],None,LB,0)),(num_uncov_edges1,node(vertices[0],None,LB,1))]
    heapq.heapify(pqueue)

    while pqueue and ((time.time() - start_time) < cutoff):

        # get the most promising decision node with the least number of uncovered edges, then the highest lower bound
        num_uncov_edges,Dnode = heapq.heappop(pqueue)

        parent = Dnode.parent
        node_id = Dnode.id   
            
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
        
        # find V_prime from Cprime
        V_prime = Find_Vprime(G,Cprime)
        # construct G_prime from V_prime
        G_prime = Gprime(G, V_prime)
        
        # find the vertex with the highest degree in G_prime that has not been explored
        new_node_id = find_next(V_prime, G_prime, explored)
        

        if not new_node_id:
            continue # prune it

        
        # Branch 1: add vertex(new_node_id) to the cover set

        cover_size = len(Cprime) + 1  
        new_cover = Cprime.union(set([new_node_id]))
    
        G_prime = G_prime.remove_vertex(new_node_id,inplace = True)
        V_prime = G_prime.get_vertices()
   
        if not V_prime:
            # solution found
            if cover_size < opt_num:
                # update solution
                opt_num = cover_size
                opt_cover = new_cover 
                # print('Optimal:', opt_num)
                continue

        
        LowerBound = cover_size + approx2(G_prime)//2

        if LowerBound < opt_num:
            new_node1 = node(new_node_id, Dnode, LowerBound,1)
            heapq.heappush(pqueue, (G_prime.num_edges, new_node1))
    

        # Branch 2: don't add vertex(new_node_id) to the cover set
        if Dnode.lb < opt_num:
            new_node0 = node(new_node_id, Dnode, Dnode.lb,0)
            heapq.heappush(pqueue, (num_uncov_edges, new_node0))

    return opt_num, opt_cover


def main(graph, algo, cutoff, seed):
    G = parse_edges(graph)
    graph_name = graph.split('/')[-1].split('.')[0]
    sol_file = "_".join([graph_name, algo, str(cutoff)]) + '.sol'
    trace_file = "_".join([graph_name, algo, str(cutoff)]) + '.trace'
    output_dir = '../output/'

    start_time = time.time()

    if algo == 'BnB':
        num_vc_nodes, vc = Branch_and_Bound(G, start_time, cutoff)


    total_time = round((time.time() - start_time), 5)
    print('BnB Algo Runtime: ' + str(total_time))

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, sol_file), 'w') as f:
        f.write(str(num_vc_nodes) + "\n")
        f.write(','.join([str(n) for n in sorted(vc)]))
    f.close()

    with open(os.path.join(output_dir, trace_file), 'w') as f:
        f.write(' '.join([str(total_time), str(num_vc_nodes)]))
    f.close()

# Run as executable from terminal
if __name__ == '__main__':
    #parse arguments in the following format: python code/approx.py -inst DATA/jazz.graph -alg approx -time 600 -seed 30
    parser = argparse.ArgumentParser(description='Run algorithm with specified parameters')
    parser.add_argument('-inst', type=str, required=True, help='graph file')
    parser.add_argument('-alg', type=str, required=True, help='algorithm to use')
    parser.add_argument('-time', type=float, default=600, required=False, help='runtime cutoff for algorithm')
    parser.add_argument('-seed', type=int, default=30, required=False, help='random seed for algorithm')
    args = parser.parse_args()

    main(args.inst, args.alg, args.time, args.seed)

        