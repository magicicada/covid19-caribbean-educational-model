"""
Runs the simulation once based on the files provided
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

from src.simulation import Simulation


# HERE BEGINS SETUP FROM FILES
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
# HERE ENDS SETUP FROM FILES



# HERE BEGINS EXPERIMENT RUNNING 

# Setup of things to consider in experiment runs
# tracing efficiencies: given a positive-tested individual $w$, the
# the probability  for each non-household contact of $w$ that they will be traced and isolated
# That is, at 0.0 no non-household contacts of positive cases are isolated, at 0.5 we expect half of them to be, etc.
tracing_efficiencies = [0.0, 0.5, 0.8]

# Testing styles:
# All testing styles should only test 'testable' people - that is those in compartments S, E, A, I 
# - 'highest_degree' tests the highest-degree testable people (i.e. those with largest number of contacts)
# - 'attribute_distrib' allocates tests to testable first-year vs testable upper-year according to a distribution spec given as a dictionary
# - None: does uniformly random testing of testable individuals
# - 'No_test' does no testing at all
# styles =  ['highest_degree', 'attribute_distrib', None, 'No_test']
styles=['No_test', 'No_test']
# The stylesWords are used as labels in figure titles. 
stylesWords = {'highest_degree':'highest degree', 'attribute_distrib':'testing only first-years', None: 'uniformly random testing',
               'alternate_null':'alternative uniform random', 'No_test': 'No testing'}
# ['powerlaw_cluster', 'regular', 'geometric']:

#  Figure plotting wrangling 
plt.clf()
fig, axs = plt.subplots(len(styles), len(tracing_efficiencies), figsize=(15, 15))
axes = plt.gca()
max_y = 5000
axes.set_ylim([0, max_y])

# Setting the type of graph we'll construct (there are several other less-structured options)
graph_type = 'education_layered'

# household and activity group size distributions
# the format here is 'year_specification':{size:frequency, size:frequency, ...}
# so  household_size_distribution = {'first':{10:0.5, 5:0.5}, 'upper':{4:0.5, 2:0.5}}
# means that 50% of first-year households are of size 10 and 50% are of size 5 (NOTE - this is not the same as 50% of first-years living in 10-person households)
household_size_distribution = {'first':{10:0.5, 5:0.5}, 'upper':{4:0.5, 2:0.5}}
number_activity_groups=500
activity_size_distribution={'first':{25:0.5, 10:0.5}, 'upper':{10:0.5, 5:0.5}}

#  setting test capacity: 0.1 is enough to test 10% of the population every day 
testProb = 0.1

# We're going to experiment for each testing style, for each tracing efficiency
for i in range(len(styles)):
    for j in range(len(tracing_efficiencies)):

        tracing_efficiency = tracing_efficiencies[j]
        test_style=styles[i]

        # For our attribute-testing we allocation between first and upper years.
        #  e.g. test_distrib={'first':1.0, 'upper':0.0} means all testing is allocation to first-years
        # test_distrib={'first':0.25, 'upper':0.75} means we allocation one quarter of our tests to first-years, three quarters to upper years
        test_distrib={'first':1.0, 'upper':0.0}
        
        
        # setting up the graph for simulation
        simulation.create_graph(graph_type, household_size_distribution=household_size_distribution, number_activity_groups=number_activity_groups, activity_size_distribution=activity_size_distribution)
        total = len(simulation.graph.graph.nodes())
        # run the simulation
        results = simulation.run_multiple(args.number_of_runs, testProb=testProb, false_positive=0.0, prob_trace_contact=tracing_efficiency,
                                          test_style=test_style, test_prob=test_distrib)
        
        
        #  adding some combination lines we might want to plot         
        results['A+I'] =  [x + y for x, y in zip(results['A'], results['I'])]
        results['A+I+T_P'] = [x + y for x, y in zip(results['A+I'], results['T_P'])]
        results['S+T_S'] = [x + y for x, y in zip(results['S'], results['T_S'])]
        results['S+T_S+E'] = [x + y for x, y in zip(results['S+T_S'], results['E'])]
        results['Cum_Cases'] = [total-x for x in results['S+T_S+E']]
        # print(results)
        
        #  lines_of_interest will set which lines to plot in the output figures
        # lines_of_interest = ['Cum_Cases'] plots only the cumulative cases
        # lines_of_interest=['R', 'A+I', 'T_S', 'T_P', 'A+I+T_P']
        lines_of_interest = ['Cum_Cases']
        
        # plotting the lines       
        for line  in lines_of_interest:
                axs[i, j].set_ylim([0, max_y])
                axs[i, j].plot(range(len(results[line])), results[line], label=str(line))
                print(line + str(results[line]))
        axs[i, j].legend()
        axs[i, j].set_ylim([0, max_y])
        axs[i, j].set_title('Testing strategy ' + str(stylesWords[test_style]) +  "\n Tracing efficiency: " + str(tracing_efficiency))
        # plt.savefig('model_output_graph-' +graph_type + "_testingProb-" + str(testProb) + "tracing_eff-" + str(tracing_efficiency) + '.png')
        
        
        # saving a file with the trajectories in
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
            
            
# saving the grid of plots 
axes = plt.gca()
axes.set_ylim([0,max_y])
# this is an entirely ad-hoc filename - you'll want to change it each time you do new runs or it will over-write 
plt.savefig('model_output_few_extras_more_balanced_edgeweights_' + graph_type+ '.png')
