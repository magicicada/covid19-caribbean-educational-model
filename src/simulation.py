import json
import pickle

import numpy as np
from datetime import datetime

from src.graph import Graph
from src.model import Model
from src.params import Params


def get_percentile(list_of_lists, what_percent):
#         assume all lists in the list of lists are of the same length
        overall_perc = []
        length = len(list_of_lists[0])
        for position in range(length):
            this_pos = []
            for this_list in list_of_lists:
                this_pos.append(this_list[position])
            this_pos = sorted(this_pos)
            perc_pos = int(round(what_percent*(len(this_pos)-1), 0))
            # print(perc_pos)
            overall_perc.append(this_pos[perc_pos])
        return overall_perc
        

class Simulation:
    """Wrapper class for managing parameters and graph and running simulations

    params : src.Params
        Parameters holding object
    graph : src.Graph
        Graph holding object
    model: src.Model
        Model of the network
    new_graph_per_run : bool, optional
        If true, builds new graph for each single simulation run,
        by default False
    load_graph_filename : str, optional
        If not empty, loads graph each run, by default ""
    verbose : bool, optional
        Print results of each simulation to console, by default False
    """

    def __init__(self,
                 infection_data_filename,
                 community_data_filename,
                 simulation_config_filename,
                 new_graph_per_run=False,
                 load_graph_filename="",
                 verbose=False):

        # set up simulation parameters
        params = Params()
        params.set_infection_data_from_file(infection_data_filename)
        params.set_community_data_from_file(community_data_filename)

        # set up the base graph configuration
        graph = Graph()
        graph.set_simulation_config_from_file(simulation_config_filename)

        self.params = params
        self.graph = graph
        self.model = None

        # build new graph for each run by default
        self.new_graph_per_run = new_graph_per_run
        self.load_graph_filename = load_graph_filename
        self.verbose = verbose

    def set_app_input_from_file(self, filename):
        """Load the input data from the app from a JSON file

        Parameters
        ----------
        filename : str
            Path to the JSON file
        """

        with open(filename, "r") as f:
            app_input_data = json.loads(f.read())

        self.params.set_app_input(
            community=app_input_data["community"],
            behaviours=app_input_data["behaviours"],
        )

    def create_graph(self, graph_type, edges_per_vert=2, household_size_distribution = {'first':{10:1.0}, 'upper':{3:1.0}}, number_activity_groups=1000, activity_size_distribution={'first':{5:1.0}, 'upper':{5:1.0}}):
        """Create graph based on the supplied parameters

        Raises
        ------
        Exception
            Throws exception when population size or infection rate
            is missing
        """

        msg = []
        if self.params.population_size is None:
            msg.append("the community has not been set")
        if self.params.generic_infection is None:
            msg.append("the infection data has not been set")

        if msg != []:
            raise Exception("Cannot create the graph: {}".format(", ".join(msg)))

        self.graph.create_graph(
            age_structure=self.params.age_structure,
            infection_rate=self.params.generic_infection,
            graph_type=graph_type,
            edgesPerVert = edges_per_vert,
            household_size_distribution=household_size_distribution,
            number_activity_groups=number_activity_groups,
            activity_size_distribution=activity_size_distribution
            # test_style=None,
            # attribute_for_test='year',
            # test_prob={'first':0.1, 'upper':0.1}
            )

    def save_graph_to_file(self, filename):
        """Save current graph object to a file

        Parameters
        ----------
        filename : str
            Filename of the file to be saved
        """

        with open(filename, "wb") as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)

    def load_graph_from_file(self, filename):
        """Load graph from a file and validate against current graph configuration

        Parameters
        ----------
        filename : str
            Filename of the saved graph object

        Raises
        ------
        Exception
            Throws exception when the loaded graph configuration differs from
            the current graph configuration
        """

        with open(filename, "rb") as f:
            loaded_graph = pickle.load(f)

        # ensure that loaded graph confirm to the graph configuration
        valid_config = loaded_graph.graph_config == self.graph.graph_config
        valid_time = loaded_graph.time_horizon == self.graph.time_horizon
        valid_infected = loaded_graph.num_infected == self.graph.num_infected
        valid_population = len(loaded_graph.graph.nodes) == self.params.population_size
        valid_infection_rate = loaded_graph.infection_rate == self.params.generic_infection

        if not(valid_config and valid_time and valid_infected and valid_population and valid_infection_rate):
            raise Exception("The graph to be loaded does not adhere to the graph configuration")

        self.graph = loaded_graph

    def run_single(self, testProb=0.1, false_positive=0.023, prob_trace_contact=0.0, test_style=None, attribute_for_test='year', test_prob={'first':0.1, 'upper':0.1}, schedule_denom =1, proportion_symptom =0.5):
        """Run a single simulation

        Returns
        -------
        dict
            Results of the simulation
        """
        print("In single run, test distribution is " +str(test_prob))
        if self.new_graph_per_run:
            if self.load_graph_filename:
                self.load_graph_from_file(self.load_graph_filename)
            else:
                self.create_graph()

        model = Model(self.params, self.graph,  self.verbose)

        model.basic_simulation(testProb=testProb, false_positive=false_positive, prob_trace_contact=prob_trace_contact, test_style=test_style, attribute_for_test=attribute_for_test, test_prob=test_prob, schedule_denom = schedule_denom)

        return model.get_results()
    
    
    
    def run_multiple(self, n, testProb=0.1, false_positive=0.023, prob_trace_contact=0.0, test_style=None, attribute_for_test='year', test_prob={'first':0.1, 'upper':0.1}, schedule_denom = 1, save_string = "trials_result", proportion_symptom = 0.5):
        """Run multiple simulations and return an averaged result

        Parameters
        ----------
        n : int
            Number of multiple simulation runs

        Returns
        -------
        dict
            Dictionary with averaged results of multiple runs
            of a simulation
        """
        now = datetime.today().isoformat()
        save_string = save_string + "_" + str(now) + "_"


        # run simulations and collect results
        all_results = [self.run_single(testProb=testProb, false_positive=false_positive, prob_trace_contact=prob_trace_contact, test_style=test_style, attribute_for_test=attribute_for_test, test_prob=test_prob, schedule_denom = schedule_denom, proportion_symptom = proportion_symptom) for _ in range(n)]
        
        # with open("state_" + save_string + ".json", 'w') as fout:
        #     json.dump(all_results , fout)
        
        
        all_repro_numbers = []
        for result in all_results:
            all_repro_numbers.append(result['repro_number'])
            
        # with open("reproductive_" + save_string + ".json", 'w') as fout:
        #     json.dump(all_repro_numbers, fout)
        
        
        all_states = list(Model.STATES.keys()) + ['cum_cases']
        
        
        # get averaged results
        averaged = {}
        for state in all_states:
            state_lists = [state_dict[state] for state_dict in all_results]
            averaged[state] = [int(np.round(np.mean(item))) for item in zip(*state_lists)]
            
           
        top = {}
        bottom = {}
        for state in all_states:
            state_lists = sorted([state_dict[state] for state_dict in all_results])
            length = len(state_lists)
            top[state] =  get_percentile(state_lists, 0.975)
            bottom[state] = get_percentile(state_lists, 0.025)
        

        # ensure number of people in each state is equal to
        # the population size at each timestep, otherwise
        # add/remove extra from biggest state
        for timestep in range(self.graph.time_horizon // 7):
            total = sum([averaged[state][timestep] for state in Model.STATES])
            if total != self.params.population_size:
                states_at_timestep = {state: averaged[state][timestep] for state in Model.STATES}
                biggest_state = max(states_at_timestep, key=states_at_timestep.get)
                averaged[biggest_state][timestep] += self.params.population_size - total
        
        print("\n\n\n\n ALL REPRO NUMBERS")
        print(all_repro_numbers)
        print("\n\n\n")
        
        #  we want a time series of the mean, top, bottom of the reproductive numbers
        averaged['repro_rate'] = get_percentile(all_repro_numbers, 0.5)
        top['repro_rate'] = get_percentile(all_repro_numbers, 0.975)
        bottom['repro_rate'] = get_percentile(all_repro_numbers, 0.025)

        
        
        return (averaged, top, bottom)
