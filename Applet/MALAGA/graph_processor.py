import pandas as pd
import networkx as nx
import pygraphviz


def modify_dot_data(dot_data: str) -> str:
    """
    Modify the DOT data by adding attributes for graph visualization.

    Parameters:
    - dot_data (str): The original DOT data as a string.
    - overlap_scale (float): The overlap scale for nodes.

    Returns:
    - str: The modified DOT data.
    """
    modified_dot_data = dot_data.replace(
        "{",
        f"{{\n    overlap = false;\n    node [shape=plaintext fontname=\"Arial\"];",
    )
    return modified_dot_data


class GraphProcessor:
    DELIMITER = ';'
    HEADER = None

    def __init__(self):
        self.graph = nx.DiGraph()

    def __init__(self, csv_file: str):
        # Initialize the graph and DOT data during the instantiation
        self.dot_data = self._generate_graph_and_dot_data(csv_file)

    def _generate_graph_and_dot_data(self, csv_file: str):
        data = self.read_csv(csv_file)
        self.create_networkx_graph(data)
        dot_data = self.generate_dot_data(modify=True)
        return dot_data

    def read_csv(self, file: str) -> pd.DataFrame:
        """
        Read a CSV file and return a Pandas DataFrame.

        Parameters:
        - file (str): The path to the CSV file.

        Returns:
        - pd.DataFrame: The DataFrame containing the CSV data.
        """
        try:
            df = pd.read_csv(file, delimiter=self.DELIMITER, header=self.HEADER)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{file}' does not exist.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file '{file}' is empty.")
        except pd.errors.ParserError:
            raise ValueError(f"Error parsing the CSV file '{file}'. Check the file format.")

    def create_networkx_graph(self, data: pd.DataFrame) -> None:
        """
        Create a NetworkX DiGraph from a Pandas DataFrame.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the CSV data.
        """
        self.graph = nx.DiGraph()

        for index, row in data.iterrows():
            node = row.iloc[0]
            self.graph.add_node(node)
            for col in row[1:]:
                if pd.notna(col):
                    self.graph.add_edge(data.iloc[int(col) - 1][0], node)
                    #self.graph.add_edge(node, data.iloc[int(col) - 1][0])

    def convert_to_graphviz(self) -> 'pygraphviz.AGraph':
        """
        Convert the internal NetworkX DiGraph to a Graphviz AGraph.

        Returns:
        - 'pygraphviz.AGraph': The Graphviz AGraph.
        """
        return nx.nx_agraph.to_agraph(self.graph)


    def generate_dot_data(self, modify: bool = False) -> str:
        """
        Generate DOT data for the contained NetworkX DiGraph.

        Parameters:
        - modify (bool): Whether to apply modifications to the DOT data.

        Returns:
        - str: The DOT data as a string.
        """
        dot_data = nx.nx_agraph.to_agraph(self.graph).to_string()

        if modify:
            dot_data = modify_dot_data(dot_data)

        return dot_data

