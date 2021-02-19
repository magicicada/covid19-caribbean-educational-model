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

styles = ['household_schedule', 'No_test']
# ['household_schedule',None, 'No_test']

plt.clf()
# plt.figure(figsize=(20, 4))
# fig, axs = plt.subplots(len(styles), len(tracing_efficiencies), figsize=(20, 20))
# axes = plt.gca()
fig, (axes, ax2) = plt.subplots(1, 2, figsize=(15, 5))


max_y = 3000
axes.set_ylim([0, max_y])

graph_type = 'education_layered'
overall_household = {1: 0.151, 2:0.105, 3:0.074, 4:0.114, 5:0.115, 6:0.130, 7:0.094, 8:0.117, 9:0.036, 10:0.029, 11:0.014, 12:0.012, 13:0.003, 14:0.003, 15:0.001, 16: 0.001, 18:0.001}
# {5:0.5, 10:0.5}
# {1: 0.151, 2:0.105, 3:0.074, 4:0.114, 5:0.115, 6:0.130, 7:0.094, 8:0.117, 9:0.036, 10:0.029, 11:0.014, 12:0.012, 13:0.003, 14:0.003, 15:0.001, 16: 0.001, 18:0.001}
overall_activity = {50:0.01, 10:0.5, 5:0.49}
# {9:0.01, 10:0.98, 4:0.01}
# {50:0.02, 10:0.5, 5:0.48}
household_size_distribution = {'first': overall_household, 'upper':overall_household}
number_activity_groups= 7000
activity_size_distribution={'first':overall_activity, 'upper':overall_activity}
sympt_from_data = 0.802

true_x = [0, 7, 14, 21, 28, 35, 42]
true_y = [34, 188, 344, 422, 492, 726, 778]

num_graphs = 50

for i in range(num_graphs):
    simulation.create_graph(graph_type, household_size_distribution=household_size_distribution, number_activity_groups=number_activity_groups, activity_size_distribution=activity_size_distribution)
    
    for i in range(len(styles)):
        # for j in range(len(tracing_efficiencies)):
        if styles[i] == 'household_schedule':
            denoms = [2, 1]
        else:
            denoms = [1]
        for denom in denoms:
            testProb= 1/14
            tracing_efficiency = 0.5
            test_style=styles[i]
    
            test_distrib={'first':0, 'upper':1.0}
            
            test_style = styles[i]
            frequency = 'never'
            if test_style == 'household_schedule':
                if denom == 1:
                    frequency = 'weekly'
                if denom == 2:
                    frequency = 'biweekly'
            if test_style == None:
                frequency == 'biweekly'
            save_string = 'trial-result_' + str(styles[i]) + "_" + frequency + "_" + str(i)
            
            total = len(simulation.graph.graph.nodes())
            # run the simulation
            (results, top, bottom) = simulation.run_multiple(args.number_of_runs, testProb=testProb, false_positive=0.0, prob_trace_contact=tracing_efficiency,
                                              test_style=test_style, test_prob=test_distrib, schedule_denom = denom, proportion_symptom=sympt_from_data, save_string = save_string)
            
            
            # output_name_this = str(test_style) + "_" + str(denom) + "_results.json"
            # json_str_results = json.dumps(results)
            # with open(output_name_this, "w") as f:
            #     f.write(json_str_results)
            #     
            # output_name_this = str(test_style) + "_" + str(denom) + "_top.json"
            # json_str_top = json.dumps(top)
            # with open(output_name_this, "w") as f:
            #     f.write(json_str_top)
            #     
            # output_name_this = str(test_style) + "_" + str(denom) + "_bottom.json"
            # json_str_bottom = json.dumps(bottom)
            # with open(output_name_this, "w") as f:
            #     f.write(json_str_bottom)
                
            
            # results['A+I'] =  [x + y for x, y in zip(results['A'], results['I'])]
            # results['A+I+T_P'] = [x + y for x, y in zip(results['A+I'], results['T_P'])]
            # results['S+T_S'] = [x + y for x, y in zip(results['S'], results['T_S'])]
            # top['S+T_S'] = [x + y for x, y in zip(top['S'], top['T_S'])]
            # results['Cum_Cases'] = [total-x for x in results['S+T_S']]
            # top['Cum_Cases'] = [total-x for x in top['S+T_S']]
    
            # lines_of_interest = ['cum_cases']
            # line_name = ''
            # test_style_string = ''
            # 
            # if test_style == 'household_schedule':
            #     if denom == 1:
            #         test_style_string = 'Weekly household testing'
            #         test_style_colour = 'blue'
            #     if denom == 2:
            #         test_style_string = 'Weekly half-household testing'
            #         test_style_colour = 'orange'
            # elif test_style == 'No_test':
            #     test_style_string = 'No asymptomatic testing'
            #     test_style_colour = 'red'
            # elif test_style == None:
            #     test_style_string = 'Random asymptomatic testing every two weeks'
            #     test_style_colour = 'grey'
            # 
            # for line  in lines_of_interest:
            #         axes.set_ylim([0, max_y])
            #         axes.plot(range(len(results[line])), results[line], label=test_style_string, color = test_style_colour)
            #         axes.fill_between(range(len(top[line])), top[line], y2 = bottom[line], color = test_style_colour, alpha = 0.2)
            #         
            #         print(line + str(results[line]))
            #         print(line + str(top[line]))
            # # axes.plot(true_x, true_y, color = 'black', label='Data')
            # axes.legend()
            # axes.set_ylim([0, max_y])
            # axes.set(xlabel = 'Day')
            # axes.set(ylabel = 'Number of cumulative cases')
            # axes.set_title('Cumulative cases over time under differing testing strategies')
            # # plt.savefig('model_output_graph-' +graph_type + "_testingProb-" + str(testProb) + "tracing_eff-" + str(tracing_efficiency) + '.png')
            # 
            # repro_name = 'repro_rate'
            # repro_med = results[repro_name]
            # ax2.plot(range(len(repro_med)), repro_med, label=test_style_string, color = test_style_colour)
            # ax2.fill_between(range(len(top[repro_name])), top[repro_name], y2 = bottom[repro_name], color = test_style_colour, alpha = 0.2)
            
            
            
            
            
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
    # axes = plt.gca()
    # axes.set_ylim([0,max_y])
    # axes.plot(true_x, true_y, color = 'black', label='Data')
    # plt.savefig('forCambridge_' + graph_type+ '.png')
