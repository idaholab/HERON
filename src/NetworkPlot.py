import networkx as nx
import matplotlib.pyplot as plt


class NetworkPlot:
    def __init__(self, components):
        self._components = components
        self._resources = set()
        self._producers_and_consumers = set()
        self._capacities = {}
        self._edges = []
        self._graph = nx.DiGraph()
        self._node_positions = None
        self._find_nodes_and_edges()
        self._plot_graph()
        self._build_table()

    def _find_nodes_and_edges(self):
        """ Iterates over components to determine nodes and their associated directional edges.
        @ In, None
        @ Out, None
        """
        for c in self._components:
            self._producers_and_consumers.add(c.name)

            rsc_in = c.get_inputs()
            for ri in rsc_in:
                self._resources.add(ri)
                self._graph.add_edge(ri, c.name)
                self._edges.append((ri, c.name))
            
            rsc_out = c.get_outputs()
            for ro in rsc_out:
                self._resources.add(ro)
                self._graph.add_edge(c.name, ro)
                self._edges.append((c.name, ro))
    
    def _get_component_capacities(self):
        for c in self._components:
            self._capacities[c.name] = c.get_capacity_var()
    
    def _build_table(self):
        """
        Table should have two major sections: economic info and optimization parameters
        
        Economic info:
          - Cash flows (just the names?)
          - Lifetime?
        
        Optimization settings:
          - dispatch (fixed, independent, dependent)
          - optimized, swept, or fixed?
          - capacity (optimization bounds, sweep values, or fixed value)
        """
        col_labels = ['Dispatchable?', 'Governed?']
        cell_text = []
        row_labels = []

        for c in self._components:
            row_labels.append(c.name)
            cell_text.append([c.is_dispatchable(), c.is_governed()])

        plt.table(cell_text, rowLabels=row_labels, colLabels=col_labels, loc='bottom')

    def _plot_graph(self):
        """ Plots and formats the graph
        @ In, None
        @ Out, None
        """
        pc_options = {  # TODO make this something that can be done dynamically
            "node_size": 1000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 1
        }
        rsc_options = {
            "node_size": 1500,
            "node_color": "yellow",
            "edgecolors": "black",
            "linewidths": 1
        }
        label_options = {
            "font_size": 8,
        }
        edge_options = {
            'edge_color': 'black',
            "width": 2,
            'arrows': True,
            'arrowsize': 40
        }

        if self._node_positions is None:
            self._node_positions = nx.spring_layout(self._graph)
        pos = self._node_positions

        nx.draw_networkx_nodes(self._graph, pos, nodelist=list(self._resources), **rsc_options)
        nx.draw_networkx_nodes(self._graph, pos, nodelist=list(self._producers_and_consumers), **pc_options)
        nx.draw_networkx_edges(self._graph, pos, **edge_options)
        nx.draw_networkx_labels(self._graph, pos, **label_options)
        # nx.draw_networkx(self._graph)
    
    def show(self):
        plt.show()

    def save(self, filename):
        plt.savefig(filename)
