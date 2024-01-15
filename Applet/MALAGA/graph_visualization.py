import networkx as nx
from pygraphviz import AGraph
from svgpathtools import parse_path
from bs4 import BeautifulSoup
import itertools as it
import pandas as pd
from typing import List, Dict, Tuple

def count_crossings(paths: List[str]) -> int:
    """
    Count the number of crossings between SVG paths.

    Args:
        paths (List[str]): List of SVG path strings.

    Returns:
        int: Number of crossings between paths.
    """
    svg_paths = [parse_path(path) for path in paths]
    crossings = 0
    for a, b in it.combinations(svg_paths, 2):
        if a.intersect(b, justonemode=True):
            crossings += 1
    return crossings


def manhattan_distance(pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
    """
    Calculate the Manhattan distance between two positions.

    Args:
        pos1 (Dict[str, float]): First position with 'x' and 'y' coordinates.
        pos2 (Dict[str, float]): Second position with 'x' and 'y' coordinates.

    Returns:
        float: Manhattan distance between pos1 and pos2.
    """
    return abs(pos1['x'] - pos2['x']) + abs(pos1['y'] - pos2['y'])

class GraphVisualizator:
    def __init__(self, graph_processor, prog: str = 'dot'):
        """
        Initialize the GraphVisualizator.

        Args:
            graph_processor: An instance of the graph processor class.
            prog (str): Graph layout program (default is 'dot').
        """
        self.graph_processor = graph_processor
        self.prog = prog
        self.svg = self.generate_svg()
        self.node_positions = self.find_positions()
        self.distance_matrix = self.calculate_distance_matrix()
        self.node_levels = self.calculate_levels()
        self.dummy_nodes = self.calculate_dummy_nodes()
        self.crossings = self.count_colliding_edges()

    def generate_svg(self) -> str:
        """
        Generate SVG representation of the graph.

        Returns:
            str: SVG representation of the graph.
        """
        dot_data = self.graph_processor.dot_data
        dot_data = dot_data.replace("node [shape=plaintext fontname=\"Arial\"];", "")
        val = AGraph(dot_data).draw(format='svg', prog=self.prog)
        return val

    def change_algorithm(self, new_prog: str) -> None:
        """
        Change the layout algorithm for the graph visualization.

        Args:
            new_prog (str): New Graphviz layout .
        """
        self.prog = new_prog
        self.svg = self.generate_svg()
        self.node_positions = self.find_positions()
        self.distance_matrix = self.calculate_distance_matrix()
        self.node_levels = self.calculate_levels()
        self.dummy_nodes = self.calculate_dummy_nodes()
        self.crossings = self.count_colliding_edges()

    def count_colliding_edges(self) -> int:
        """
        Count the number of colliding edges in the graph.

        Returns:
            int: Number of colliding edges.
        """
        soup = BeautifulSoup(self.svg, 'xml')
        edge_paths = [edge_path.find('path').get('d') for edge_path in soup.find_all('g', class_='edge') if
                      edge_path.find('path')]
        num = count_crossings(edge_paths)
        return num

    def find_positions(self) -> Dict[str, Dict[str, float]]:
        """
        Find positions of nodes in the SVG representation.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of node positions.
        """
        soup = BeautifulSoup(self.svg, 'xml')
        positions = {}

        # Find all nodes in the SVG
        nodes = soup.find_all('g', class_='node')

        # Extract x and y coordinates for each node
        for node in nodes:
            title = node.find('title').text.strip()
            x = float(node.find('ellipse')['cx'])
            y = float(node.find('ellipse')['cy'])
            positions[title] = {'x': x, 'y': y}

        return positions

    def calculate_distance_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate the Manhattan distance matrix between nodes.

        Returns:
            Dict[str, Dict[str, float]]: Distance matrix between nodes.
        """
        nodes = list(self.node_positions.keys())
        distance_matrix = {node: {} for node in nodes}

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                pos1 = self.node_positions[node1]
                pos2 = self.node_positions[node2]
                distance_matrix[node1][node2] = manhattan_distance(pos1, pos2)

        return distance_matrix

    def dataframe_distance_matrix(self) -> pd.DataFrame:
        """
        Convert the distance matrix to a Pandas DataFrame.

        Returns:
            pd.DataFrame: Distance matrix as a DataFrame.
        """
        nodes = list(self.distance_matrix.keys())
        df = pd.DataFrame(self.distance_matrix, index=nodes)
        return df

    def calculate_levels(self) -> Dict[str, int]:
        """
        Calculate the levels of nodes based on their y-coordinates.

        Returns:
            Dict[str, int]: Dictionary mapping nodes to levels.
        """
        levels = {}
        for node, data in self.node_positions.items():
            y = data['y']
            if y not in levels:
                levels[y] = [node]
            else:
                levels[y].append(node)

        levels = dict(sorted(levels.items()))

        node_levels = {}

        level = 1
        for list in levels.values():
            for item in list:
                node_levels[item] = level
            level += 1
        return node_levels


    def dummy_nodes_from_edges(self, edge1: str, edge2: str) -> int:
        """
        Calculate the number of dummy nodes needed between two edges, as the number of needed dummy nodes that
        should be created at each level that the edge cross.

        Args:
            edge1 (str): First edge.
            edge2 (str): Second edge.

        Returns:
            int: Number of dummy nodes needed.
        """
        return abs(self.node_levels[edge2] - self.node_levels[edge1]) - 1

    def calculate_dummy_nodes(self) -> int:
        """
        Calculate the total number of dummy nodes needed in the graph.

        Returns:
            int: Total number of dummy nodes.
        """
        return sum(self.dummy_nodes_from_edges(edge[0], edge[1]) for edge in self.graph_processor.graph.edges)
