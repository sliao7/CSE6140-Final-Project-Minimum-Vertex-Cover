import networkx as nx
import time
import os
import argparse


def parse_edges(filename):
    # parse edges from graph file to create your graph object
    # filename: string of the filename
    f = open(filename, "r")
    n_vertices, n_edges, _ = f.readline().split(' ')
    n_vertices, n_edges = int(n_vertices), int(n_edges)

    G = nx.Graph()  # create a graph

    # add edges to the graph
    for i in range(1, n_vertices + 1):
        neighbors = f.readline().rstrip().split(' ')
        for neighbor in neighbors:
            if neighbor != '':
                if int(neighbor) > i:
                    G.add_edge(i, int(neighbor))
    return G

def mdg(G, start_time, cutoff):
    degrees = dict(G.degree())
    cover = []
    max_node = max(degrees , key = degrees.get)
    while (degrees[max_node] > 0) and ((time.time() - start_time) < cutoff):
        degrees[max_node] = 0
        cover.append(max_node)
        for n in G.neighbors(max_node):
            degrees[n] -= 1
        max_node = max(degrees , key = degrees.get)

    return len(cover), cover


def matching_approx(G, start_time, cutoff):
    """
    Find approx min vertex using a maximal matching in the graph.

    A matching is a subset of edges in which no node occurs more than once.
    A maximal matching cannot add more edges and still be a matching.

    Input: a networkx undirected graph
    Returns: set of edges consisting of a maximal matching of the graph.

    """
    max_match = set()
    nodes = set()
    if (time.time() - start_time) < cutoff:
        for u, v in G.edges():
            # If the edge is uncovered add it to matching set
            # then remove neighborhood of u and v
            if u not in nodes and v not in nodes and u != v:
                max_match.add(u)
                max_match.add(v)
                nodes.add(u)
                nodes.add(v)

    return len(nodes), max_match


def gic(G, start_time, cutoff):
    cover = set()
    edges = G.number_of_edges()

    while (edges) > 0 and ((time.time() - start_time) < cutoff):
        min_node = min(dict(G.degree), key=dict(G.degree).get)
        for v in list(G.neighbors(min_node)):
            cover.add(v)  # add to vertex cover
            G.remove_node(v)  # remove neighbor node
        G.remove_node(min_node)  # remove min degree node
        edges = G.number_of_edges()

    return len(cover), cover

def main(graph, algo, cutoff, seed):
    G = parse_edges(graph)
    graph_name = graph.split('/')[-1].split('.')[0]
    sol_file = "_".join([graph_name, algo, str(cutoff)]) + '.sol'
    trace_file = "_".join([graph_name, algo, str(cutoff)]) + '.trace'
    output_dir = '../output/'

    start_time = time.time()

    if algo == 'Approx':
        num_vc_nodes, vc = mdg(G, start_time, cutoff)

    total_time = round((time.time() - start_time), 5)
    print('Approx Algo Runtime: ' + str(total_time))

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