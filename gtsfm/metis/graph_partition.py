import torch
import numpy as np
import networkx as nx
import metis
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from typing import List, Tuple


class BinaryPartitionTreeNode:
    """Represents a node in the binary partition tree."""

    def __init__(self, graph_edges: List[Tuple[int, int]], nodes: List[int]):
        self.graph_edges = graph_edges
        self.nodes = nodes
        self.left = None
        self.right = None


class BinaryPartitionTree:
    """Binary tree structure for recursive graph partitioning."""

    def __init__(self, similarity_matrix: torch.Tensor, max_levels=3):
        """
        Constructs a binary partition tree based on a similarity matrix.

        Args:
            similarity_matrix (torch.Tensor): A square similarity matrix representing the relationships between nodes.
            max_levels (int): The depth of the partition tree.
        """
        self.max_levels = max_levels
        self.G = self._build_graph(similarity_matrix)
        self.root = self._partition_graph(self.G, level=0)

    def _build_graph(self, similarity_matrix: torch.Tensor) -> nx.Graph:
        """Creates an undirected weighted graph from the similarity matrix."""
        num_nodes = similarity_matrix.shape[0]
        sim_matrix_np = similarity_matrix.numpy()
        sim_matrix_np = (sim_matrix_np + sim_matrix_np.T) / 2  # Ensure symmetry
        np.fill_diagonal(sim_matrix_np, 0)  # Remove self-loops

        G = nx.Graph()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                G.add_edge(i, j, weight=sim_matrix_np[i, j])
        return G

    def _partition_graph(self, graph: nx.Graph, level: int) -> BinaryPartitionTreeNode:
        """Recursively partitions the graph and builds the binary tree."""
        if level == self.max_levels or len(graph.nodes()) <= 1:
            return BinaryPartitionTreeNode(list(graph.edges()), list(graph.nodes()))

        edgecuts, partitions = metis.part_graph(graph, nparts=2, recursive=True)
        partition_0 = [
            node for i, node in enumerate(graph.nodes()) if partitions[i] == 0
        ]
        partition_1 = [
            node for i, node in enumerate(graph.nodes()) if partitions[i] == 1
        ]

        subgraph_0 = graph.subgraph(partition_0).copy()
        subgraph_1 = graph.subgraph(partition_1).copy()

        node = BinaryPartitionTreeNode(list(graph.edges()), list(graph.nodes()))
        node.left = self._partition_graph(subgraph_0, level + 1)
        node.right = self._partition_graph(subgraph_1, level + 1)

        return node

    def get_partition_at_level(self, level: int) -> dict:
        """
        Returns a mapping of nodes to partition IDs at a given level.

        Args:
            level (int): The partition level.

        Returns:
            dict: A mapping {node_id: partition_id}.
        """
        partition_map = {}
        self._collect_partitions(
            self.root,
            partition_map,
            target_level=level,
            current_level=0,
            partition_id=0,
        )
        return partition_map

    def _collect_partitions(
        self,
        node: BinaryPartitionTreeNode,
        partition_map: dict,
        target_level: int,
        current_level: int,
        partition_id: int,
    ):
        """Recursively assigns nodes to partitions at a given level."""
        if node is None:
            return
        if current_level == target_level or (node.left is None and node.right is None):
            for n in node.nodes:
                partition_map[n] = partition_id
        else:
            self._collect_partitions(
                node.left,
                partition_map,
                target_level,
                current_level + 1,
                partition_id * 2,
            )
            self._collect_partitions(
                node.right,
                partition_map,
                target_level,
                current_level + 1,
                partition_id * 2 + 1,
            )

    def get_leaf_partitions(self) -> List[List[Tuple[int, int]]]:
        """
        Returns all leaf node partitions.

        Returns:
            List[List[Tuple[int, int]]]: A list of leaf partitions, each containing a list of edges (node_a, node_b).
        """
        leaf_partitions = []

        def collect_leaves(node: BinaryPartitionTreeNode):
            if node is None:
                return
            if node.left is None and node.right is None:  # Leaf node
                leaf_partitions.append(node.graph_edges)
            else:
                collect_leaves(node.left)
                collect_leaves(node.right)

        collect_leaves(self.root)
        return leaf_partitions

    def visualize_graph(self):
        """Visualizes the graph using Plotly 3D, with intra-partition edges matching node color."""

        def plot_level(level, title):
            node_to_partition = self.get_partition_at_level(level)
            unique_partitions = sorted(set(node_to_partition.values()))

            # Assign colors to partitions
            partition_to_color = {
                p: pc.qualitative.Plotly[i % len(pc.qualitative.Plotly)]
                for i, p in enumerate(unique_partitions)
            }

            # Generate 3D node positions
            pos = nx.spring_layout(self.G, dim=3, seed=42)

            # Full graph plot
            fig = go.Figure()

            for edge in self.G.edges():
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]

                partition_a = node_to_partition[edge[0]]
                partition_b = node_to_partition[edge[1]]

                # Edge color logic
                if partition_a == partition_b:
                    edge_color = partition_to_color[
                        partition_a
                    ]  # Intra-partition edge (same as node color)
                else:
                    edge_color = "#D3D3D3"  # Inter-partition edge (light gray)

                fig.add_trace(
                    go.Scatter3d(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        z=[z0, z1, None],
                        line=dict(width=2, color=edge_color),
                        hoverinfo="none",
                        mode="lines",
                    )
                )

            # Add nodes
            for part in unique_partitions:
                partition_nodes = [
                    node for node in self.G.nodes() if node_to_partition[node] == part
                ]
                part_node_x = [pos[node][0] for node in partition_nodes]
                part_node_y = [pos[node][1] for node in partition_nodes]
                part_node_z = [pos[node][2] for node in partition_nodes]

                fig.add_trace(
                    go.Scatter3d(
                        x=part_node_x,
                        y=part_node_y,
                        z=part_node_z,
                        mode="markers",
                        marker=dict(
                            size=10, color=partition_to_color[part], opacity=0.9
                        ),
                        hoverinfo="text",
                    )
                )

            fig.update_layout(
                title=title, showlegend=False, margin=dict(b=0, l=0, r=0, t=40)
            )
            fig.show()

            # Display all partitions in one page with edges
            if level > 0:
                cols = min(4, len(unique_partitions))
                rows = (len(unique_partitions) + cols - 1) // cols
                subplot_fig = make_subplots(
                    rows=rows,
                    cols=cols,
                    subplot_titles=[f"Partition {p}" for p in unique_partitions],
                    specs=[[{"type": "scatter3d"}] * cols] * rows,
                )

                for i, part in enumerate(unique_partitions):
                    partition_nodes = [
                        node
                        for node in self.G.nodes()
                        if node_to_partition[node] == part
                    ]
                    partition_edges = [
                        (u, v)
                        for u, v in self.G.edges()
                        if u in partition_nodes and v in partition_nodes
                    ]

                    part_node_x = [pos[node][0] for node in partition_nodes]
                    part_node_y = [pos[node][1] for node in partition_nodes]
                    part_node_z = [pos[node][2] for node in partition_nodes]

                    part_edge_x, part_edge_y, part_edge_z = [], [], []
                    for edge in partition_edges:
                        x0, y0, z0 = pos[edge[0]]
                        x1, y1, z1 = pos[edge[1]]
                        part_edge_x.extend([x0, x1, None])
                        part_edge_y.extend([y0, y1, None])
                        part_edge_z.extend([z0, z1, None])

                    edge_trace = go.Scatter3d(
                        x=part_edge_x,
                        y=part_edge_y,
                        z=part_edge_z,
                        line=dict(width=1, color=partition_to_color[part]),
                        hoverinfo="none",
                        mode="lines",
                    )

                    partition_trace = go.Scatter3d(
                        x=part_node_x,
                        y=part_node_y,
                        z=part_node_z,
                        mode="markers",
                        marker=dict(
                            size=10, color=partition_to_color[part], opacity=0.9
                        ),
                        hoverinfo="text",
                    )

                    row, col = divmod(i, cols)
                    subplot_fig.add_trace(edge_trace, row=row + 1, col=col + 1)
                    subplot_fig.add_trace(partition_trace, row=row + 1, col=col + 1)

                subplot_fig.update_layout(title=f"All Partitions at Level {level}")
                subplot_fig.show()

        plot_level(3, "Level 3: Eight Partitions")
        plot_level(2, "Level 2: Four Partitions")
        plot_level(1, "Level 1: Two Partitions")


# Example Usage
similarity_matrix = torch.rand(20, 20)
similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
similarity_matrix.fill_diagonal_(0)

binary_tree = BinaryPartitionTree(similarity_matrix, max_levels=3)
binary_tree.visualize_graph()

leaf_partitions = binary_tree.get_leaf_partitions()

# Print the output format: List[List[Tuple[int, int]]]
for i, partition in enumerate(leaf_partitions):
    print(f"Leaf Partition {i}: {partition}")
