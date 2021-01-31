import itertools
import collections
import json
import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

### Notes on networks library
# The implementation is currently fully based on networkx.
# A mixed implementation of networkx (small world graph generation) and
# python-igraph (rest of the operations) performed slightly worse in terms
# of runtime but much better in terms of memory usage.
# If memory usage is a concern, using python-igraph is recommended.

class Graph:
    """Graph and related data

    graph : networkx.classes.graph.Graph
        Graph modelling the population
    time_horizon : int
        Number of iterations of the simulation
    num_infected : int
        Number of initial infections
    graph_config : dict
        Dictionary with graph constructor configurations
    infection_rate : float
        Infection probability
    close_nodes_arr : numpy.ndarray
        Array indicating whether a node is close to another node
    """

    def __init__(self):
        # Graph and config
        self.graph = None
        self.time_horizon = None
        self.num_infected = None
        self.graph_config = None
        self.infection_rate = None

        # Simulation structures
        self.close_nodes_arr = None

    def draw_graph(self, filename):
        """Save a graph visualisation to a file

        The method is intended for development purposes only.
        If a graph has more than 100 nodes the method is not executed.

        Parameters
        ----------
        filename : str
            Path to where to save the resulting visualisation
        """

        if len(self.graph.nodes()) > 100:
            print("WARNING - [draw_graph] more than 100 nodes in the graph.")
            # return

        layout = nx.spring_layout(self.graph)
        weights = [1 + self.graph[u][v]["weight"]
                   if self.graph[u][v] else 1
                   for u, v in self.graph.edges()]

        nx.draw(self.graph, layout,
                with_labels=False,
                edges=self.graph.edges(),
                width=weights,
                node_size=30)

        plt.savefig(filename)


    def set_simulation_config_from_file(self, filename):
        """Load simulation config from a JSON file

        Parameters
        ----------
        filename : str
            Path to the JSON file
        """

        with open(filename, "r") as f:
            simulation_config = json.loads(f.read())

        self.time_horizon = simulation_config["time_horizon"]
        self.num_infected = simulation_config["num_infected"]
        self.graph_config = simulation_config["graph_config"]


    def create_graph(self, age_structure, infection_rate, graph_type='regular', edgesPerVert=4, household_size_distribution = {'first':{10:1.0}, 'upper':{3:1.0}}, number_activity_groups=1000, activity_size_distribution={'first':{5:1.0}, 'upper':{5:1.0}}):
                     # 
        """Create a random AMENDED geometric, etc

        Parameters
        ----------
        age_structure : dict
            Dictionary indicating number of people in given age group
        infection_rate : float
            Infection probability
        """

        self.infection_rate = infection_rate
        
        if graph_type == 'geometric':
           self._set_random_geometric_graph(age_structure=age_structure, edgesPerVert=edgesPerVert)
        elif graph_type == 'regular':
           self._set_random_regular_graph(age_structure=age_structure, edgesPerVert=edgesPerVert)
        elif graph_type == 'powerlaw_cluster':
            self._set_powerlaw_cluster(age_structure=age_structure, edgesPerVert=edgesPerVert)
        elif graph_type == 'education_layered':
            secondary_ratio = 0.1
            house_weight = 0.02*1.5
            self._set_household_groupings(age_structure=age_structure, household_size_distribution=household_size_distribution, household_edge_weight=house_weight)
            self._add_secondary_groupings(number_activity_groups, activity_size_distribution=activity_size_distribution, activity_edge_weight = secondary_ratio*house_weight)
            # first_year_extras = 3
            # upper_year_extras = 20
            # nodes = list(self.graph.nodes())
            # 
            # self._add_extra_contacts(self.graph, nodes, 5)
            
            # self._add_extra_contacts(self.graph, nodes, 0.001)
            # first_years = [x for x,y in self.graph.nodes(data=True) if y['year']=='first']
            # upper_years = [x for x,y in self.graph.nodes(data=True) if y['year']=='upper']
            # # # 
            # self._add_extra_contacts(self.graph, first_years, first_year_extras, extra_edge_weight=0.05)
            # self._add_extra_contacts(self.graph, upper_years, upper_year_extras)
            # 
            # degrees = list(sorted(self.graph.degree, key=lambda x: x[1], reverse=True))
            # for node,degree in degrees[:100]:
            #     print(str(node) + "  " + self.graph.nodes[node]['year'] + "  " + str(degree))
            
        else:
            self._set_navigable_small_world_graph(
                age_structure=age_structure,
                short_connection_diameter=self.graph_config["params"]["short_connection_diameter"],
                long_connection_diameter=self.graph_config["params"]["long_connection_diameter"],
                decay=self.graph_config["params"]["decay"])
        
        # print('number of edges: ' + str(len(self.graph.edges())))
        # print('clustering coefficient '  + str(nx.average_clustering(self.graph)))
        # self.draw_graph(graph_type + ".trial.pdf")
        # self.plot_degree_distrib()
        
        
    def set_group_interaction_edges(self, behaviour, node, other_nodes,
                                    group_size):
        """Set edges in the graph based on specified group interaction

        Parameters
        ----------
        behaviour : str
            Name of the behaviour
        node : int
            Label of the node
        other_nodes : list
            List of other nodes
        group_size : int
            Number of nodes in a group
        """

        # if behaviour == "food_shopping":
        #     # swap other nodes for only close nodes
        #     other_nodes = self.close_nodes_arr[node].nonzero()[0].tolist()

        # set group size
        size = group_size if group_size <= len(other_nodes) else len(other_nodes)

        # choose new group members randomly
        new_group = random.sample(other_nodes,
                                  k=size)

        # get new edges to add
        if behaviour == "food_shopping":
            new_edges = [(node, neighbour) for neighbour in new_group]
        else:
            # add the node considered
            new_group.append(node)
            # add edges so that the new group makes a connected subgraph
            new_edges = itertools.combinations(new_group, 2)

        # add new edges
        list(map(self._add_new_interaction_edge, new_edges))

    def remove_group_interaction_edges(self):
        """Remove interaction edges from the graph"""

        # remove new edges
        for (u, v) in self.graph.edges:
            if "interaction_edge" in self.graph[u][v]:
                self.graph.remove_edge(u, v)



    def _add_new_interaction_edge(self, pair):
        """Add a new interaction edge if it is not already existing

        Parameters
        ----------
        pair : tuple
            Tuple in form (node, new neighbor node)
        """

        if not(self.graph.has_edge(*pair)):
            self.graph.add_edge(*pair, weight=self.infection_rate,
                                interaction_edge=True)
    
    
    def choose_from_distrib(self, distrib):
        curr_sum = 0
        max_sum = random.random()
        for value in distrib:
            curr_sum += distrib[value]
            if max_sum <= curr_sum:
                return value
            
        return None
    
    
    def _sub_household(self, num_people, graph, year_string, household_id, household_size_distribution, household_edge_weight):
        people_thus_far = 0
        while people_thus_far < num_people:
            generate_household_size = self.choose_from_distrib(household_size_distribution)
            # print('generating a household of size ' + str(generate_household_size))
            
            if people_thus_far + generate_household_size >= num_people:
                generate_household_size = num_people - people_thus_far
                # Make a clique of new people to add
            for i in range(generate_household_size):
                graph.add_node((household_id, i), household=household_id, year=year_string)
            for i in range(generate_household_size):
                for j in range(i+1, generate_household_size):
                    graph.add_edge((household_id, i), (household_id, j), weight=household_edge_weight)
            household_id = household_id +1
            people_thus_far = people_thus_far + generate_household_size
        return household_id
    
    def _set_household_groupings(self, age_structure, household_size_distribution, household_edge_weight=0.05):
        # print(household_size_distribution)
        num_people = sum(age_structure.values())
        people_thus_far = 0
        household_id = 0
        graph = nx.Graph()
        
        # first years
        # 
        num_first_year = 0
        # math.floor(num_people/4)
        # print('number of first-years is ' + str(num_first_year))
        # household_id = self._sub_household(num_first_year, graph, 'first', household_id, household_size_distribution['first'], household_edge_weight=household_edge_weight)
        
        # upper years
        num_upper_year = num_people - num_first_year
        household_id = self._sub_household(num_upper_year, graph, 'upper', household_id, household_size_distribution['upper'], household_edge_weight=household_edge_weight)
        
        # print('\n\n people thus far ' + str(people_thus_far))
        
        self.graph = graph
        self._set_ages_uniform('(10, 19)')
    
    
    def _add_secondary_groupings(self, number_activity_groups, activity_size_distribution, activity_edge_weight = 0.01):
#         note: hard-coded to year classes just now
        for cat in activity_size_distribution:
            activity_size_distribution_this = activity_size_distribution[cat]
            graph = self.graph
            these_nodes = []
            for k,v in graph.nodes(data=True):
                if v['year'] == cat:
                    these_nodes.append(k)
            if len(these_nodes) >0:
                for i in range(number_activity_groups):
                    group_size = self.choose_from_distrib(activity_size_distribution_this)
                    
                    group_members = list(random.sample(these_nodes, group_size))
                    for i in range(len(group_members)):
                        for j in range(i+1, len(group_members)):
                            if (group_members[i], group_members[j]) not in graph.edges():
                                graph.add_edge(group_members[i], group_members[j], weight=activity_edge_weight)
                        
    
    def _add_extra_contacts(self, graph, vertex_set, num_extras_per, extra_edge_weight=0.01):
        n = len(vertex_set)
        total = n*num_extras_per
        for _ in range(total):
            i = random.randint(0,n-1)
            j  = random.randint(0,n-1)
            graph.add_edge(vertex_set[i], vertex_set[j], weight=extra_edge_weight)
            
#         TODO - change this to uniformly at random, ER style
        # pref_att_graph = nx.barabasi_albert_graph(n, p)
        # pref_att_graph = nx.fast_gnp_random_graph(n, p)
        # nodes_list = list(pref_att_graph.nodes())
        # for i in range(len(nodes_list)):
        #     for j in range(i+1, len(nodes_list)):
        #         if (i, j) in pref_att_graph.edges() and (vertex_set[i], vertex_set[i]) not in graph.edges():
        #             graph.add_edge(vertex_set[i], vertex_set[j], weight=extra_edge_weight)
        # 
        
    
    
    
    def _set_powerlaw_cluster(self, age_structure, edgesPerVert=4):
        num_people = sum(age_structure.values())
        graph = nx.powerlaw_cluster_graph(num_people, edgesPerVert, 0.3)
        
       # set the graph and add infection attribute
        self.graph = graph
        self._set_generic_weights()
        self._set_ages(age_structure)        
    
    
    def _set_random_regular_graph(self, age_structure, edgesPerVert=4):
        num_people = sum(age_structure.values())
        print(num_people)        
        graph = nx.random_regular_graph(edgesPerVert*2, num_people)
        
       # set the graph and add infection attribute
        self.graph = graph
        self._set_generic_weights()
        self._set_ages(age_structure)
    
    def _set_random_geometric_graph(self, age_structure, edgesPerVert=4):
        num_people = sum(age_structure.values())
        # expected degree is n*pi*r^2
        # so if degree is edgesPerVert*2, then r should be root(edgesPerVert*2/(n*pi))   - I'm going to lazily use 3.14 as pi
        radius = math.sqrt(edgesPerVert*2/(num_people*3.14))
        
        graph = nx.random_geometric_graph(num_people, radius)
        
       # set the graph and add infection attribute
        self.graph = graph
        self._set_generic_weights()
        self._set_ages(age_structure)
        
    
    def _set_navigable_small_world_graph(self, age_structure,
                                         short_connection_diameter,
                                         long_connection_diameter,
                                         decay):
        """Create non-directional Navigable Small World graph

        Parameters
        ----------
        age_structure : dict
            Dictionary indicating number of people in given age group
        short_connection_diameter : int
            Diameter of short connections
        long_connection_diameter : int
            Diameter of long connections
        decay : float
            Decay exponent
        """

        # get total number of people
        num_people = sum(age_structure.values())

        # since the graph constructor takes n as side of the grid
        # and returns n**2 nodes, take ceiling of square root
        # and remove the extra nodes later
        root_num_people = math.ceil(math.sqrt(num_people))

        # construct the graph
        digraph = nx.navigable_small_world_graph(root_num_people,
                                                 short_connection_diameter,
                                                 long_connection_diameter,
                                                 decay,
                                                 dim=2)

        # remove the extra unnecessary nodes
        num_extra = root_num_people**2 - num_people
        extra_nodes = random.sample(list(digraph.nodes),
                                    k=num_extra)
        digraph.remove_nodes_from(extra_nodes)

        # ignore directional edges, move grid location to attribute
        # and relabel with integer values
        graph = nx.convert_node_labels_to_integers(
            digraph.to_undirected(),
            label_attribute="location")

        # remove possible selfloops
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # set the graph and add infection attribute
        self.graph = graph
        self._set_generic_weights()
        self._set_ages(age_structure)

        # precompute close nodes
        self._set_close_nodes_arr()

    def _set_close_nodes_arr(self):
        """Compute and store list of close nodes for each node in the graph

        Nodes are close if they are within Chebyshev distance specified by
        the threshold (treats diagonal connections the same as the adjacent)"""

        threshold = self.graph_config["closeness_threshold"]

        n_nodes = len(self.graph.nodes)
        node_arr = np.zeros((n_nodes, n_nodes))
        for node in self.graph:

            node_location = self.graph.nodes[node]["location"]

            # go over all nodes except node in question
            for other_node in self.graph.nodes:
                if node != other_node:

                    other_location = self.graph.nodes[other_node]["location"]

                    # calculate Chebyshev distance
                    # (includes diagonal neighbours)
                    chebyshev_dist = self._calculate_chebyshev_dist(other_location,
                                                                    node_location)

                    # save node if it is close
                    if chebyshev_dist <= threshold:
                        node_arr[node][other_node] = 1
        self.close_nodes_arr = node_arr

    def _calculate_chebyshev_dist(self, location_a, location_b):
        """Calculate Chebyshev distance

        Parameters
        ----------
        location_a : tuple
            Tuple in form (x, y)
        location_b : tuple
            Tuple in form (x, y)

        Returns
        -------
        int
            Chebyshev distance
        """

        return max(abs(location_a[0] - location_b[0]),
                   abs(location_a[1] - location_b[1]))

    def _set_generic_weights(self):
        """Add generic infection probability to each edge in the graph"""

        nx.set_edge_attributes(self.graph, self.infection_rate, "weight")

    def _set_ages(self, age_structure):
        """Set age of a node based on the age structure of the population

        Parameters
        ----------
        age_structure : dict
            Dictionary indicating number of people in given age group
        """

        age_group_list = []
        for age_group, n in age_structure.items():
            age_group_list += [age_group] * n

        random.shuffle(age_group_list)
        age_group_dict = {i: age_group_list[i] for i in range(len(age_group_list))}
        nx.set_node_attributes(self.graph, age_group_dict, "age_group")

    
    def _set_ages_uniform(self, single_age):
        """Set age of a node based on the age structure of the population

        Parameters
        ----------
        age_structure : dict
            Dictionary indicating number of people in given age group
        """
        age_group_dict = {}
        for node in self.graph.nodes():
            age_group_dict[node] = single_age

        nx.set_node_attributes(self.graph, age_group_dict, "age_group")
        
    
    def plot_degree_distrib(self, filename='degree_distrib.png'):
        G = self.graph
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
        print(degree_sequence)
        # degreeCount = collections.Counter(degree_sequence)
        # deg, cnt = zip(*degreeCount.items())
        # print(deg)
        # print(cnt)
        
        # fig, ax = plt.subplots()
        # plt.bar(deg, cnt, width=0.80, color="b")
        # 
        # plt.title("Degree Histogram")
        # plt.ylabel("Count")
        # plt.xlabel("Degree")
        # ax.set_xticks([d + 0.4 for d in deg])
        # ax.set_xticklabels(deg)
        # plt.savefig(filename)