import BnB
import SA
import approx
import GA

import time
import os
import argparse
import random

opt_cutoff = {'karate':14, 'football':94, 'jazz':158, 'email':594, 'delaunay_n10':703,'netscience':899, 'power':2203,'as-22july06':3303,'hep-th':3926,'star2':4542,'star':6902}


def main(graph, algo, cutoff, seed):
    random.seed(seed)

    graph_name = graph.split('/')[-1].split('.')[0]

    sol_file = "_".join([graph_name, algo, str(cutoff), str(seed)]) + '.sol'
    trace_file = "_".join([graph_name, algo, str(cutoff), str(seed)]) + '.trace'
    output_dir = './output/' #'./{}_output/'.format(algo)

    start_time = time.time()

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fo = open(os.path.join(output_dir, trace_file), 'w')

    if algo == 'BnB':
        if graph_name not in opt_cutoff:
            return

        G = BnB.parse_edges(graph)
        num_vc_nodes, vc = BnB.Branch_and_Bound(
            G, start_time, cutoff, fo, opt_cutoff[graph_name], seed)
        fo.close()

        total_time = round((time.time() - start_time), 5)
        print('BnB Algo Runtime: ' + str(total_time))

        with open(os.path.join(output_dir, sol_file), 'w') as f:
            f.write(str(num_vc_nodes) + "\n")
            f.write(','.join([str(n) for n in sorted(vc)]))
        f.close()

    if algo == 'SA':
        sa_obj = SA.SA()
        G, nV, nE = sa_obj.parse_edges(graph)

        G_init = G.copy()
        sol = sa_obj.initial_solution(
            G=G_init, fo=fo, start_time=start_time, cutoff=cutoff, input_file=graph)
        final_solution = sa_obj.simulate_annealing(
            G, fo, sol, cutoff, nV, start_time, graph, opt_cutoff.get(graph_name, 10))
        fo.close()
        print('SA Solution: ({}) {}'.format(
            len(final_solution), final_solution))

        total_time = round((time.time() - start_time), 5)
        print('SA Runtime (s): {}'.format(total_time))

        with open(os.path.join(output_dir, sol_file), 'w') as f:
            f.write(str(nV) + "\n")
            f.write(','.join([str(n) for n in sorted(final_solution)]))
        f.close()

    if algo == 'approx':
        G = approx.parse_edges(graph)
        num_vc_nodes, vc = approx.mdg(G, start_time, cutoff)

        total_time = round((time.time() - start_time), 5)
        print('Approx Algo Runtime: ' + str(total_time))

        with open(os.path.join(output_dir, sol_file), 'w') as f:
            f.write(str(num_vc_nodes) + "\n")
            f.write(','.join([str(n) for n in sorted(vc)]))
        f.close()

        with open(os.path.join(output_dir, trace_file), 'w') as f:
            f.write(' '.join([str(total_time), str(num_vc_nodes)]))
        f.close()

    if algo == 'GA':
        runner = GA.manual_runner(graph)

        pop_size = 3 * len(runner.graph.vert_dict)
        best_ind = runner.run_test(MU=pop_size,
                                   MUTPB=0.07, CXPB=0.8, NGEN=2000, verbose=False, cutoff=cutoff, trace_path=os.path.join(output_dir, trace_file), start=start_time)
        total_time = round((time.time() - start_time), 5)
        print('GA Runtime: ' + str(total_time))
        num_nodes = sum(best_ind)
        solution_vertices = runner.sorted_vertex_ids(best_ind)

        with open(os.path.join(output_dir, sol_file), 'w') as f:
            f.write(str(num_nodes) + "\n")
            f.write(','.join([str(n) for n in sorted(solution_vertices)]))

        with open(os.path.join(output_dir, trace_file), 'a') as f:
            f.write(','.join([str(total_time), str(num_nodes)]))


# Run as executable from terminal
if __name__ == '__main__':
    # parse arguments in the following format:
    # python Python/main.py -inst DATA/email.graph -alg approx -time 600 -seed 30

    parser = argparse.ArgumentParser(
        description='Run algorithm with specified parameters')
    parser.add_argument('-inst', type=str, required=True, help='graph file')
    parser.add_argument('-alg', type=str, required=True,
                        help='algorithm to use')
    parser.add_argument('-time', type=float, default=600,
                        required=False, help='runtime cutoff for algorithm')
    parser.add_argument('-seed', type=int, default=30,
                        required=False, help='random seed for algorithm')
    args = parser.parse_args()

    main(args.inst, args.alg, args.time, args.seed)
