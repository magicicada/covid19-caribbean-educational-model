"""
Runs the simulation once based on the files provided
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

from src.simulation import Simulation

parser = argparse.ArgumentParser(
    description="Runs the simulation once based on the files provided")

parser.add_argument("-i", "--input_spec_file", required=True, type=str,
                    metavar="file",
                    help="""JSON file specifying locations of data files required for the specification.
                            See input files specification in data/README.md for details.
                            'real' defaults to data/real/input-spec.json;
                            'sample' defaults to data/sample/sample-input-spec.json""")

parser.add_argument("-o", "--output_filename", required=False, type=str,
                    metavar="file", default="data/output/sample-output.json",
                    help="""Filename where the results should be saved.
                            data/output/sample-output.json by default.""")

parser.add_argument("-n", "--number_of_runs", type=int,
                    metavar="int", default=10,
                    help="""Number of runs of the simulation. If bigger than 1,
                            the simulation results are averaged across the runs.
                            10 by deafult.""")

parser.add_argument("-lg", "--load_graph_dir", type=str,
                    metavar="str", default="",
                    help="""When supplied, attempts to load graph from a file
                            {community_name}-graph.pkl in specified
                            directory.""")

parser.add_argument("-ng", "--new_graph", action="store_true",
                    help="""Construct a graph on per simulation basis instead of once
                            per run.""")

parser.add_argument("-q", "--quiet", action="store_true",
                    help="Do not print results to console.")

parser.add_argument("-gp", "--graph_plot_filename", type=str,
                    metavar="file",
                    help="When supplied, saves PNG graph visualisation to given filename.")

parser.add_argument("-dtp", "--doubling_time_plot_filename", type=str,
                    metavar="file",
                    help="When supplied, saves PNG doubling time plot to given filename.")

args = parser.parse_args()

start_time = datetime.now()

# define input files
if args.input_spec_file == "real":
    input_file_spec = "data/real/input-spec.json"
elif args.input_spec_file == "sample":
    input_file_spec = "data/sample/sample-input-spec.json"
else:
    input_file_spec = args.input_spec_file

with open(input_file_spec, "r") as f:
    input_files = json.loads(f.read())
    app_input_filename = input_files["app_input_filename"]
    community_data_filename = input_files["community_data_filename"]
    infection_data_filename = input_files["infection_data_filename"]
    simulation_config_filename = input_files["simulation_config_filename"]

simulation = Simulation(infection_data_filename,
                        community_data_filename,
                        simulation_config_filename,
                        new_graph_per_run=args.new_graph,
                        verbose=not(args.quiet))
simulation.set_app_input_from_file(app_input_filename)

testProbs = [0.01, 0.05, 0.1]
tracing_efficiencies = [0.0, 0.5, 1.0]
for graph_type in ['education_layered']:
# ['powerlaw_cluster', 'regular', 'geometric']:
    plt.clf()
    fig, axs = plt.subplots(len(testProbs), len(tracing_efficiencies), figsize=(15, 15))
    
    household_size_distribution = {4:0.2, 10:0.2, 2:0.5, 20:0.1}
    number_activity_groups=1000
    activity_size_distribution={5:0.5, 10:0.5}
    
    # plt.subplots_adjust(top=1.2)
    for i in range(len(testProbs)):
        for j in range(len(tracing_efficiencies)):
            testProb=testProbs[i]
            tracing_efficiency = tracing_efficiencies[j]
            
            simulation.create_graph(graph_type, edges_per_vert=10, household_size_distribution=household_size_distribution, number_activity_groups=number_activity_groups, activity_size_distribution=activity_size_distribution)
            
            # run the simulation
            results = simulation.run_multiple(args.number_of_runs, testProb=testProb, false_positive=0.01, prob_trace_contact=tracing_efficiency)
            
            print(results)
            lines_of_interest=['S', 'E', 'I', 'A', 'R', 'T_S', 'T_P']
            
            for line in results:
                print(results[line])
                if line in lines_of_interest:
                    axs[i, j].plot(range(len(results[line])), results[line], label=str(line))
            # axs[i, j].legend()
            axs[i, j].set_title('Testing every :' + str(1/testProb) + " days, \n Tracing efficiency: " + str(tracing_efficiency))
            # plt.savefig('model_output_graph-' +graph_type + "_testingProb-" + str(testProb) + "tracing_eff-" + str(tracing_efficiency) + '.png')
            
            if args.output_filename:
                json_str = json.dumps(results)
            
                with open(args.output_filename, "w") as f:
                    f.write(json_str)
            # 
            # if args.graph_plot_filename:
            #     simulation.graph.draw_graph(args.graph_plot_filename)
            
            if args.doubling_time_plot_filename:
                simulation.model.plot_doubling_time(args.doubling_time_plot_filename)
            
            if not args.quiet:
                print()
                print(datetime.now() - start_time)
    plt.savefig('model_output_' + graph_type+ '.png')
