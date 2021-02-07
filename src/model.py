import random
from collections import Counter
from functools import partial
import matplotlib.pyplot as plt
import networkx as nx
from time import gmtime, strftime 


def generate_household_testing_schedule(denom_test_week, proportion_consenting, graph):
    schedule = {}
    week_sched = {}
    for i in range(denom_test_week):
        week_sched[i] = []
    
    house_lists = {}
    house_lists_consented = {}
    for guy in graph:
        house = graph.nodes[guy]['household']
        if house not in house_lists:
            house_lists[house] = [guy]
        else:
            house_lists[house].append(guy)
    for house in house_lists:
        consent_total = int(round(len(house_lists[house])*proportion_consenting, 0))
        for i in range(consent_total):
            week_sched[i%denom_test_week].append(house_lists[house][i])
    return week_sched

def _make_daily_schedule(schedule, graph, period_length=7):
  # daily_sched will be a dict of dicts, indexed like daily_sched[week][day]
  daily_sched = {}


  households = []
  for guy in graph:
    house = graph.nodes[guy]['household']
    households.append(house)
  households = list(set(households))

  # each household gets a day of the week at random
  house_to_day = {}
  for house in households:
    house_to_day[house] = random.randrange(period_length)

  # each person gets a (week, day).  The day comes from the household
  # the week comes from the schedule
  for week in schedule:
    if week not in daily_sched:
      daily_sched[week] = {}
    these_guys = schedule[week]
    for guy in these_guys:
      house = graph.nodes[guy]['household']
      this_day = house_to_day[house]
      this_week = week
      if this_day not in daily_sched[week]:
        daily_sched[week][this_day] = []
      daily_sched[week][this_day].append(guy)     
  return daily_sched

class Model:
    """Base model for simulating COVID-19 spread in Jamaica

    Parameters
    ----------
    params : src.Params
        Parameters holding object
    graph : src.Graph
        Graph holding object
    console_log : bool, optional
        Log execution output, by default False
    STATES : dict
        Dictionary with state names
    """

    STATES = {"S": "Susceptible",
              "E": "Exposed",
              "A": "Presymptomatic",
              "A_T": 'Asymptomatic',
              "I": "Symptomatic",
              "H": "Hospitalised",
              "D": "Dead",
              "R": "Recovered",
              "T_P": "Tested_Isolated_Positive",
              "T_S": "Tested_Isolated_Susceptible"}

    def __init__(self, params, graph, console_log=False):
        self.params = params
        self.graph = graph
        self.infection_tree = nx.DiGraph()
        self.console_log = console_log
        self.overall_r = 0
        


    def basic_simulation(self, testProb=0.1, false_positive=0.0, prob_trace_contact=0.0, test_style=None, attribute_for_test='year', test_prob={'first':0.25, 'upper':0.75}, schedule_denom = 1, proportion_symptom = 0.5):
        """Run the simulation"""
        
        if test_style == 'household_schedule':
#             generate a household schedule - right now hackily hard-coded
           denom = schedule_denom
           proportion_consenting = 1.0
           res = generate_household_testing_schedule(denom, proportion_consenting, self.graph.graph)
           schedule = _make_daily_schedule(res, self.graph.graph, period_length=7)


        self.params.behaviours_dict = self.params._convert_behaviours_to_dict()
        self.infection_tree = nx.DiGraph()

        # choose a random set of initially infected
        infected = random.sample(list(self.graph.graph),
                                 k=self.graph.num_infected)

        self._make_states_dict()
        step_zero_states_dict = self.states_dict[0]
        for vertex in infected:
            step_zero_states_dict[vertex] = "A"

        nodes = self.graph.graph.nodes
        will_have_sympts = {}
        thresh_for_sympt = proportion_symptom
        for node in nodes:
            will_have_sympts[node] = random.random() < thresh_for_sympt
                
        
        num_tests = int(testProb*len(nodes))
#             we need to generate a number of tests for each category
        num_by_attr = {}
        for cat in test_prob:
            num_by_attr[cat] = test_prob[cat]*num_tests

        for time in range(self.graph.time_horizon):
            self.curr_time = time

            # use map for better performance
            # use list to force map evaluation
            # list(map(self._add_interactions, nodes))
            list(map(self._do_progression, nodes))
            list(map(self._do_infection, nodes))
            
            # for node in nodes:
            #         self._do_testing(node, 0, 0, prob_trace_contact)
            for node in self.graph.graph.nodes():
                if will_have_sympts[node]:
                    self._do_symptomatic_testing( node, false_positive, prob_trace_contact)
                
            if test_style == 'highest_degree':
                highest_degree_group = list(sorted(self.graph.graph.degree, key=lambda x: x[1], reverse=True))
                i = 0
                for node in highest_degree_group:
                        if self.testable(node[0]):
                            self._do_testing(node[0], testProb=1.0, false_positive=false_positive, prob_trace_contact=prob_trace_contact)
                            i=i+1
                        if i >= num_tests:
                            break
            elif test_style == 'household_schedule':

                  period = 7
                  # schedule = [period][day]
                  denom = max(schedule.keys()) + 1
                  which_day = time%period
                  which_period = int(time/period)%denom
                  guys_for_test = schedule[which_period][which_day]

                  i = 0
                  for node in guys_for_test:
                    if self.testable(node):
                       self._do_testing(node, testProb=1.0, false_positive=false_positive, prob_trace_contact=prob_trace_contact)
                       i = i+1


            elif test_style == 'attribute_distrib':
                self._do_strategic_testing_category(self.graph.graph, attribute_for_test, num_by_attr, test_prob_trace_contact=prob_trace_contact)
            elif test_style == 'alternate_null':
                random_nodes = []
                for node in self.graph.graph.nodes():
                    random_nodes.append(node)
                random.shuffle(random_nodes)
                i=0
                for node in random_nodes:
                        if self.testable(node):
                            self._do_testing(node, testProb=1.0, false_positive=false_positive, prob_trace_contact=prob_trace_contact)
                            i=i+1
                        if i >= num_tests:
                            break
            elif test_style == None:
                # theseNodes = choose
                # print('I am doing no testing')
                # num_tested = 0
                for node in nodes:
                     if self.testable(node):
                         self._do_testing(node, testProb=testProb, false_positive=false_positive, prob_trace_contact=prob_trace_contact)
                #             num_tested = num_tested+1
            # else:
            #     for node in nodes:
            #         self._do_testing(node, 0, 0, prob_trace_contact)
            # list(map(self._do_testing, nodes))

            # self._remove_interactions()

        if self.console_log:
            self.print_state_counts(self.graph.time_horizon)
        # plt.clf()
        # degrees = [self.infection_tree.out_degree(n) for n in self.infection_tree.nodes()]
        # plt.hist(degrees)
        # # nx.draw_networkx(self.infection_tree)
        # plt.savefig('tree_degree_distrib_' + str(strftime("%Y-%m-%d%H:%M:%S", gmtime())) + '.pdf')
    
    def get_reproductive_number(self):
        """Returns a float giving the mean out-degree from the infection tree, ignoring very recent infections
        

        Returns
        -------
        float
            mean out-degree from infection tree, approximation of reproductive number
        """
        
        time_cutoff = 80
        
        this_infection_tree = nx.DiGraph()
        vertices_for_inclusion = []
        out_degrees = []
        
        for (a, b) in self.infection_tree.edges():
          if self.infection_tree[a][b]['time'] < time_cutoff:
            vertices_for_inclusion.append(b)
        for node in vertices_for_inclusion:
          out_degrees.append(self.infection_tree.out_degree(node))

        print("\n\n\n all out degrees")
        print(out_degrees)
        print("\n\n\n")
        
        
        mean_out_degree = sum(out_degrees)/len(out_degrees)
        return mean_out_degree
        
    def get_results(self):
        """Returns dictionary with results of the simulation

        The results are reported every 7 steps. - AMENDED TO DAILY
        

        Returns
        -------
        dict
            Dictionary in form {state: [number of nodes in given state]} plus one entry 'repro_number': float recording out_degree of infection tree
        """
        non_case = ['S', 'E', 'T_S']

        output_dict = {key: [] for key in Model.STATES.keys()}
        output_dict['cum_cases']= []

        for step in range(1, self.graph.time_horizon + 1, 1):
            counts = self._get_state_counts(step)
            
            cum_case_count = 0
                      
            for state in output_dict:
                if state not in non_case:
                  cum_case_count = cum_case_count + counts[state]
                if state != 'cum_cases':
                  output_dict[state].append(counts[state])
            # print('appending ' + str(cum_case_count) + ' cumulative counts')
            output_dict['cum_cases'].append(cum_case_count)
            
        output_dict['repro_number'] = self.get_reproductive_number()
        return output_dict

    def print_state_counts(self, time, letter_only=False):
        """Print the summary of number of nodes in given state at given time

        Parameters
        ----------
        time : int
            Timestep
        letter_only : bool, optional
            Print the state string letter instead of the full name,
            by default False
        """

        counts = self._get_state_counts(time)

        print("Time:", time)
        for letter, name in Model.STATES.items():
            if letter_only:
                print("{:<5}{:>5}".format(letter, counts[letter]))
            else:
                print("{:<15}{:>10}".format(name, counts[letter]))
        print("-"*30)

    def plot_doubling_time(self, filename):
        """Create plot of cumulative cases and doubling time
        and save it to a file.

        Parameters
        ----------
        filename : str
            Name of the file where the figure is to be saved
        """

        # generate list of cumulative cases
        # where cases is everyone not is state "S"
        infectious_counts = []
        for timestep in range(self.graph.time_horizon):
            counts = self._get_state_counts(timestep)
            timestep_count = self.params.population_size - counts["S"]
            infectious_counts.append(timestep_count)

        # generate the list of doubling times
        # ie how many timesteps is takes for
        # number of cases to double
        double_times = []
        for ix in range(len(infectious_counts)):
            for ix_future in range(ix + 1, len(infectious_counts)):
                if infectious_counts[ix_future] >= infectious_counts[ix] * 2:
                    double_times.append(ix_future - ix)
                    break

        # since double_times will be shorter, append extra
        # None for plotting
        diff = len(infectious_counts) - len(double_times)
        if diff != 0:
            double_times += [None] * diff

        # construct the figure
        # based on https://matplotlib.org/gallery/api/two_scales.html
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("timestep")
        ax1.set_ylabel("cumulative cases", color=color)
        ax1.plot(range(len(infectious_counts)), infectious_counts, color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xticks(range(0, len(infectious_counts), 5))

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("timesteps to double", color=color)
        ax2.plot(range(len(infectious_counts)), double_times, color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_yticks(range(11))

        ax1.grid(True, axis="x")
        fig.tight_layout()
        plt.savefig(filename)

    def _make_states_dict(self):
        """Create states dictionary for timestep 0 where
        each node from graph has state 'S'
        """

        states_dict = {timestep: {} for timestep in range(self.graph.time_horizon + 1)}
        for guy in self.graph.graph:
            states_dict[0][guy] = "S"

        self.states_dict = states_dict

    def _choose_from_distrib(self, age_group, distrib):
        """Randomly choose a state based on given distribution

        Parameters
        ----------
        age_group : str
            Name of the age group
        distrib : dict
            Dictionary in form {state: {age_group: prob}}

        Returns
        -------
        str
            Randomly chosen item
        """

        curr_sum = 0
        max_sum = random.random()
        for value in distrib:
            curr_sum += distrib[value][age_group]
            if max_sum <= curr_sum:
                return value

        print(("Something has gone wrong - no next state was returned. "
               "Choosing arbitrarily."))
        print(distrib)
        return min(distrib.keys())

    def _add_interactions(self, node):
        """Add interactions to the graph based on the number of visits
        of set behaviours

        Parameters
        ----------
        node : int
            Name of node in the graph
        """

        # get the current "weekday"
        weekday = self.curr_time % 7

        # get all other nodes
        other_nodes = [other_node for other_node in self.graph.graph if other_node != node]

        # set up the week
        if weekday == 0:
            for behaviour, behaviour_dict in self.params.behaviours_dict.items():
                # randomly choose days for each interaction
                chosen_days = random.sample(range(7), k=behaviour_dict["visits"])
                self.graph.graph.nodes[node][behaviour] = {"visit_days": chosen_days}

        # do the interactions
        for behaviour, behaviour_dict in self.params.behaviours_dict.items():
            # add given behaviour if it should happen on current weekday
            if weekday in self.graph.graph.nodes[node][behaviour]["visit_days"]:

                # set the interactions
                self.graph.set_group_interaction_edges(
                    behaviour=behaviour,
                    node=node,
                    other_nodes=other_nodes,
                    group_size=behaviour_dict["num_people"]
                )

    def _remove_interactions(self):
        """Remove the added interactions from the graph"""

        self.graph.remove_group_interaction_edges()
    
    def testable(self, node):
        state = self.states_dict[self.curr_time][node]
        return state == 'S' or state == 'E' or state == 'I' or state == 'A' or state =='R'
    
    def _do_progression(self, node):
        """Progress the states based on the state transition dictionary

        Simplest sensible model: no age classes, uniform transitions
        between states, each vertex will have a state at each timestep.

        Parameters
        ----------
        node : int
            Name of node in the graph
        """

        next_time = self.curr_time + 1

        # get the state, then then possibilities
        state = self.states_dict[self.curr_time][node]
        node_age_group = self.graph.graph.nodes[node]["age_group"]

        # R, D, S stay the same (assumes recovery grants immunity)
        # E, A, I, H change based on transitions
        if state not in ["R", "D", "S"]:
            self.states_dict[next_time][node] = self._choose_from_distrib(
                                    node_age_group,
                                    self.params.state_transitions[state])
        else:
            self.states_dict[next_time][node] = state

    def _do_infection(self, node):
        """Spread the infection

        Parameters
        ----------
        node : int
            Name of node in the graph"""

        state = self.states_dict[self.curr_time][node]

        if state == "I" or state == "A" or state == "A_T":
            list(map(partial(self._infect_neighbours, node),
                     self.graph.graph.neighbors(node)))
            
         # TODO: add continued household infection after isolation
        if state == "T_P":
          # do only household infection
          list(map(partial(self._infect_household_only, node),
                     self.graph.graph.neighbors(node)))
          

    
    # def _do_strategic_testing_set(self, graph, set_of_nodes, test_prob_trace_contact=0.0):
    #     for node in set_of_nodes:
    #         self._do_testing(node, testProb=1.0, false_positive=0.0, prob_trace_contact=test_prob_trace_contact)
    
    def _do_strategic_testing_category(self, graph, attribute_for_test, test_prob, test_prob_trace_contact):
        
        for node in graph.nodes():
            self._do_testing(node, 0, 0, test_prob_trace_contact)
                
        
#         we could pre-compute this to save time
        guys_in_cats = {}
        for cat in test_prob:
            guys_in_cats[cat] = []
        for k,v in graph.nodes(data=True):
            attribute = v[attribute_for_test]
            guys_in_cats[attribute].append(k)

        for cat in guys_in_cats:
            tests_avail = test_prob[cat]
            # print('now testing category ' + str(cat) + ' with ' + str(tests_avail) + ' tests')
            random.shuffle(guys_in_cats[cat])
            
            # print('for cat ' + str(cat) + ' list of guys to test length is ' + str(len(guys_in_cats[cat])))
            i = 0
            for k in guys_in_cats[cat]:
                if self.testable(k):
                    if self._do_testing(k, testProb=1.0, false_positive=0.0, prob_trace_contact=test_prob_trace_contact):
                       i = i+1
                if i >= tests_avail:
                    break
    
    def _do_symptomatic_testing(self, node, false_positive, prob_trace_contact):
        state = self.states_dict[self.curr_time][node]
        did_test = False
        # get test result
        if state == "I":
             self._isolate_self_and_neighbours(node, prob_trace_contact=prob_trace_contact)
             self._isolate_household(node)
             return did_test
        
    
    def _do_testing(self, node, testProb, false_positive, prob_trace_contact):
        """Do the testing and isolation

        Parameters
        ----------
        node : int
            Name of node in the graph"""
        state = self.states_dict[self.curr_time][node]
        did_test = False
        # get test result
        
        thisLuck = random.random()
        
        # if state == "I":
        #     self._isolate_self_and_neighbours(node, prob_trace_contact=prob_trace_contact)
        #     self._isolate_household(node)
        #     return did_test
                
        # print(testProb)
        if thisLuck <= testProb:
            did_test = True
            if state == "I" or state == "A" or state == "A_T":
                    self._isolate_self_and_neighbours(node, prob_trace_contact=prob_trace_contact)
                    self._isolate_household(node)
            elif state == 'S':
                thisLuck = random.random()
                if thisLuck < false_positive:
                    self._isolate_self_and_neighbours(node, false_positive=True, prob_trace_contact=prob_trace_contact)
                    self._isolate_household(node)
        return did_test
        
            
#             # list(map(partial(self._infect_neighbours, node),
#             #          self.graph.graph.neighbors(node)))
# #     TODO - WORKING PLACE
#     def generate_household_testing_schedule(denom_test_week, proportion_consenting, graph):
#         schedule = {}
#         week_sched = {}
#         for i in range(denom_test_week):
#             week_sched[i] = []
#         
#         house_lists = {}
#         house_lists_consented = {}
#         for guy in graph:
#             house = graph.nodes[guy]['household']
#             if house not in house_lists:
#                 house_lists[house] = [guy]
#             else:
#                 house_lists[house].append(guy)
#         for house in house_lists:
#             consent_total = int(round(len(house_lists[house])*proportion_consenting, 0))
#             for i in range(consent_total):
#                 week_sched[i%denom_test_week].append(house_lists[house][i])
#         return week_sched
        
    
    def _isolate_household(self, node):
      # get the household
      household = self.graph.graph.nodes[node]['household']
      neighbours = self.graph.graph.neighbors(node)
      for guy in neighbours:
        if self.graph.graph.nodes[guy]['household'] == household:
            state = self.states_dict[self.curr_time][guy]
            if state == "I" or state == "A":
                self.states_dict[self.curr_time + 1][guy] = "T_P"
            elif state == 'S':
                self.states_dict[self.curr_time + 1][guy] = "T_S"
            
    def _isolate_self_and_neighbours(self, node, false_positive=False, prob_trace_contact=0.0):
        """Isolation for node and neighbours

        Parameters
        ----------
        node : int
            Label of the node
        """
        if false_positive:
            self.states_dict[self.curr_time + 1][node] = "T_S"
        else:
            self.states_dict[self.curr_time + 1][node] = "T_P"
            
        neighbours = self.graph.graph.neighbors(node)
        for guy in neighbours:
            state = self.states_dict[self.curr_time][guy]
            thisLuck  = random.random()
            if thisLuck < prob_trace_contact:
                if state == "I" or state == "A":
                    self.states_dict[self.curr_time + 1][guy] = "T_P"
                elif state == 'S':
                    self.states_dict[self.curr_time + 1][guy] = "T_S"
    
    def _infect_neighbours(self, node, neighbour):
        """Infect a neighbour node

        Parameters
        ----------
        node : int
            Label of the node
        neighbour : int
            Label of the neighbour node
        """

        if self.states_dict[self.curr_time][neighbour] == "S":
            infection_prob = self.graph.graph[node][neighbour]["weight"]
            luck = random.random()
            if luck <= infection_prob:
                self.states_dict[self.curr_time + 1][neighbour] = "E"
                self.infection_tree.add_edge(node, neighbour, time = self.curr_time )
                
    def _infect_household_only(self, node, neighbour):
        """Infect a the household members

        Parameters
        ----------
        node : int
            Label of the node
        neighbour : int
            Label of the neighbour node
        """
        this_household = self.graph.graph.nodes[node]['household']
        household_neigh = self.graph.graph.nodes[neighbour]['household']
        

        if this_household == household_neigh and self.states_dict[self.curr_time][neighbour] == "S":
            infection_prob = self.graph.graph[node][neighbour]["weight"]
            luck = random.random()
            if luck <= infection_prob:
                self.states_dict[self.curr_time + 1][neighbour] = "E"
                self.infection_tree.add_edge(node, neighbour, time = self.curr_time )

    def _get_state_counts(self, time):
        """Return the number of nodes in given state at given time

        Parameters
        ----------
        time : int
            Time step at which the states are counted

        Returns
        -------
        dict
            Dictionary in form {state: count}
        """
        return Counter(self.states_dict[time].values())
