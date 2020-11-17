# from graph import graph

from deap import base
from deap import creator
from deap import tools

import argparse

import random
import numpy as np
from time import time

import os

# read data and construct a graph object


class vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = set()

    def __str__(self):
        # for print out result
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor):

        self.adjacent.add(neighbor)

    def remove_neighbor(self, neighbor):
        if neighbor in self.adjacent:
            self.adjacent.remove(neighbor)

    def is_connected(self, neighbor):
        return neighbor in self.adjacent

    def get_connections(self):
        return self.adjacent


class graph:
    # unweighted undirected graph
    # can be connected or not
    edges = None

    def __init__(self, edge_list=False):
        self.vert_dict = {}  # vertex_id (int) : vertex
        self.num_vertices = 0
        self.num_edges = 0

        if edge_list:
            self.edges = []  # list of edge tuples

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices += 1
        new_vertex = vertex(node)
        self.vert_dict[node] = new_vertex

    def get_vertex(self, node):
        if node in self.vert_dict:
            return self.vert_dict[node]
        else:
            return None

    def add_edge(self, frm, to):
        # for new vertices
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        if not self.vert_dict[frm].is_connected(self.vert_dict[to]):
            self.num_edges += 1

        self.vert_dict[frm].add_neighbor(self.vert_dict[to])
        self.vert_dict[to].add_neighbor(self.vert_dict[frm])
        self.edges.append((frm, to))

    def remove_edge(self, frm, to):
        self.vert_dict[frm].remove_neighbor(self.vert_dict[to])
        self.vert_dict[to].remove_neighbor(self.vert_dict[frm])
        self.num_edges -= 1
        if self.edges is not None:
            self.edges = [e for e in self.edges if not e ==
                          (frm, to) and not e == (to, frm)]

    def get_vertices(self):
        # return a list of ints, the id of vertices
        return list(self.vert_dict.keys())


def main():
    data = ['6 7 0', '2 3', '1 3 4', '1 2 5', '2 5 6', '3 4', '4']
    g = graph()
    num_ver, num_edges, _ = map(int, data[0].split(' '))

    for i in range(1, num_ver + 1):
        neighbors = map(int, data[i].split(' '))
        for neighbor in neighbors:
            if neighbor > i:
                g.add_edge(i, neighbor)

    for v in g.get_vertices():
        print(v)
        print(g.get_vertex(v))

    print('Number of edges: ', g.num_edges)
    print('Number of vertices, ', g.num_vertices)


def parse_edges(filename):
    # parse edges from graph file to create your graph object
    # filename: string of the filename
    f = open(filename, "r")
    n_vertices, n_edges, _ = f.readline().strip().split(' ')
    n_vertices, n_edges = int(n_vertices), int(n_edges)

    G = graph(edge_list=True)  # create a graph

    # add edges to the graph
    for i in range(1, n_vertices+1):
        # neighbors = map(int, f.readline().strip().split(' '))
        neighbors = f.readline().rstrip().split(' ')
        for neighbor in neighbors:
            if neighbor != '':
                if int(neighbor) > i:
                    G.add_edge(int(i), int(neighbor))
    return G


class manual_runner():
    def __init__(self, fname):
        self.graph = parse_edges(fname)
        self.vertex_to_ids = {i: k for i,
                              k in enumerate(self.graph.vert_dict.keys())}

    def create_individual(self, bar=0.8):
        NUM_VERTICES = len(self.graph.get_vertices())
        return [1 if random.random() < bar else 0 for i in range(NUM_VERTICES)]

    def run_test(self, MU=50, NGEN=10, LAMBDA=100, CXPB=0.7, MUTPB=0.2, verbose=False, cutoff=600, trace_path=None, start=0):
        NUM_VERTICES = len(self.graph.get_vertices())
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        # Attribute generator
        toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, NUM_VERTICES)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)
        toolbox.register("evaluate", self.score_candidate)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutateFlip", tools.mutFlipBit,
                         indpb=(1.0/NUM_VERTICES))
        toolbox.register("mutateShuffle", tools.mutShuffleIndexes,
                         indpb=(1.0/NUM_VERTICES))
        toolbox.register("select", tools.selTournament, tournsize=3)
        # toolbox.register("select", tools.selRandom)

        s = time()
        seeds = [creator.Individual([1 for i in range(NUM_VERTICES)])]

        bar = 1.0 - (1./NUM_VERTICES)

        pop = [seeds[0]]
        # for i in range(100):
        #     pop.append(creator.Individual(
        #         [1 if c != i else 0 for c in range(NUM_VERTICES)]))
        num_individuals = max(NUM_VERTICES // 500, 100)
        for i in range(num_individuals):
            pop.append(creator.Individual(
                [1 if c != i else 0 for c in range(NUM_VERTICES)]))
        for i in range(num_individuals):
            pop.append(creator.Individual(self.create_individual()))
        # for i in range(NUM_VERTICES):
        #     pop.append(creator.Individual(self.create_individual()))
        # for m in range(MU-1):
        #     pop.append(creator.Individual(
        #         [1 if random.random() < bar else 0 for i in range(NUM_VERTICES)]))
        if verbose:
            print("Start of evolution")

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit if None not in fit else (-1,)
        if verbose:
            print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0
        overall_best = None
        stagnant = 0
        # Begin the evolution
        # while g < NGEN and time() - s < 595 and (stagnant < 1000 or True):
        while g < NGEN and time() - s < cutoff - 2 and stagnant < 1000:
            # A new generation
            g = g + 1
            if g % 10 == 0 and verbose:
                print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutateFlip(mutant)
                    del mutant.fitness.values
                # else:
                #     mutant[0] = 0
                    # toolbox.mutateFlip(mutant)
                    # del mutant.fitness.values
                elif random.random() < MUTPB:
                    toolbox.mutateShuffle(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if g % 10 == 0 and verbose:
                print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            if overall_best is None or overall_best.fitness.values[0] < max(fits):
                stagnant = 0
                total_time = round((time() - s), 2)
                overall_best = tools.selBest(pop, 1)[0]
                if trace_path is not None:
                    with open(trace_path, 'a') as f:
                        f.write(
                            ','.join([str(time() - start), str(sum(overall_best))]) + "\n")
            else:
                stagnant += 1

            if g % 10 == 0 and verbose:
                print(
                    f"  Min {min(fits):.2f} \t Max {max(fits):.2f} \t Avg {mean:.2f} \t std: {std:.2f}")

        if verbose:
            print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        if verbose:
            print("Best individual is %s, %s" %
                  (best_ind, best_ind.fitness.values))
            print("Overall Best individual is %s, %s" %
                  (overall_best, overall_best.fitness.values))
        self.best = best_ind
        self.overall_best = overall_best
        e = time()
        if verbose:
            print(f"Time: {e - s}s")

        return overall_best
        # print(f"{pop}-{stats}-{hof}")

    def sorted_vertex_ids(self, c):
        return sorted([self.vertex_to_ids[i]
                       for i in range(len(c)) if c[i] == 1])

    def score_candidate(self, c):
        # verts = set([i+1 for i in range(len(c)) if c[i] == 1])
        verts = set([self.vertex_to_ids[i]
                     for i in range(len(c)) if c[i] == 1])
        if len(c) < len(self.graph.vert_dict):
            return (-1,)
        score = 0
        total_verts = 0
        bad = 0
        for e in self.graph.edges:
            if e[0] in verts or e[1] in verts:
                score += 1
            else:
                # print(f"{e[0]}-{e[1]} - {e[0] in verts} {e[1] in verts}")
                bad += 1
        if score != len(self.graph.edges):
            # print(f"{score} vs {len(self.graph.edges)}")
            # return (-1 * bad,)
            return (-1,)
            # return (None,)
#         if score == len(self.graph.edges):
#             score *= len(self.graph.get_vertices())
#         score = score - len(verts)
        return (len(self.graph.get_vertices()) - len(verts),)


def main(graph_name, cutoff, seed, algo, verbose=False):
    random.seed(seed)
    s = time()
    graph_path = graph_name.split('/')[-1].split('.')[0]
    sol_file = "_".join([graph_path, algo, str(cutoff)]) + '.sol'

    trace_file = "_".join([graph_path, algo, str(cutoff)]) + '.trace'
    output_dir = '../output/'

    runner = manual_runner(graph_name)

    pop_size = 3 * len(runner.graph.vert_dict)
    best_ind = runner.run_test(MU=pop_size,
                               MUTPB=0.1, CXPB=0.001, NGEN=1000, verbose=verbose, cutoff=cutoff, trace_path=os.path.join(output_dir, trace_file), start=s)
    total_time = round((time() - s), 2)
    num_nodes = sum(best_ind)
    solution_vertices = runner.sorted_vertex_ids(best_ind)
    if verbose:
        print(f"Number Vertices: {sum(num_nodes)}")

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, sol_file), 'w') as f:
        f.write(str(num_nodes) + "\n")
        f.write(','.join([str(n) for n in sorted(solution_vertices)]))

    with open(os.path.join(output_dir, trace_file), 'a') as f:
        f.write(','.join([str(total_time), str(num_nodes)]))

    # print(runner.graph.edges)


# Run as executable from terminal
if __name__ == '__main__':
    # parse arguments in the following format: python code/approx.py -inst DATA/jazz.graph
    parser = argparse.ArgumentParser(
        description='Run algorithm with specified parameters')
    parser.add_argument('-inst', type=str, required=True, help='graph file')
    parser.add_argument('-time', default=600, type=float,
                        required=False, help='Cutoff running time for algorithm')
    parser.add_argument('-seed', default=1000,
                        type=int, required=False, help='Random Seed for algorithm')
    parser.add_argument('-alg', default='LS2', type=str,
                        required=False, help='Choice Algorithm')
    args = parser.parse_args()

    main(args.inst, args.time, args.seed, args.alg)
