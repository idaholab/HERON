import networkx as nx
import matplotlib.pyplot as plt


def plot_resource_network(components):
    for c in components:
        print(c, c.get_inputs(), c.get_outputs(), c.name)


class NetworkPlot:
    def __init__(self, components):
        self._components = components
        self._resources = set()
        self._producers_and_consumers = set()
        self._edges = []
        self._graph = nx.DiGraph()
        self._find_nodes_and_edges()
        self._plot_graph()

    def _find_nodes_and_edges(self):
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

    def _plot_graph(self):
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

        pos = nx.kamada_kawai_layout(self._graph)

        nx.draw_networkx_nodes(self._graph, pos, nodelist=list(self._resources), **rsc_options)
        nx.draw_networkx_nodes(self._graph, pos, nodelist=list(self._producers_and_consumers), **pc_options)
        nx.draw_networkx_edges(self._graph, pos, **edge_options)
        nx.draw_networkx_labels(self._graph, pos, **label_options)
        # nx.draw_networkx(self._graph)
    
    def show(self):
        plt.show()

    def save(self, filename):
        plt.savefig(filename)
