import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures
# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 1. Load the data
df = pd.read_csv('csv.csv')

# 2. Create a directed graph
G: nx.DiGraph = nx.DiGraph()

# 3. Define colors and shapes for different node types
node_colors = {
    'places:City': '#FF9999',       # Light red for cities
    'places:County': '#99CCFF',     # Light blue for counties
    'ctm:MusicType': '#99FF99',     # Light green for music types
    'ctm:SpecialIndependentResource': '#FFCC99'  # Light orange for resources
}

node_shapes = {
    'places:City': 'o',             # Circle for cities
    'places:County': 's',           # Square for counties
    'ctm:MusicType': '^',           # Triangle for music types
    'ctm:SpecialIndependentResource': 'd'  # Diamond for resources
}

# 4. Define colors and styles for different edge types
edge_colors = {
    '毗邻': 'blue',
    '县、区级行政单位隶属于…市级行政单位': 'red',
    '分布地域': 'green',
    '特藏资源涉及乐种': 'orange'
}

edge_styles = {
    '毗邻': 'solid',
    '县、区级行政单位隶属于…市级行政单位': 'solid',
    '分布地域': 'dashed',
    '特藏资源涉及乐种': 'dotted'
}

# 5. Add nodes and edges to the graph
for _, row in df.iterrows():
    source = row['sourceLabel']
    source_type = row['sourceNodeType']
    target = row['targetLabel']
    target_type = row['targetNodeType']
    relation = row['relationType']
    
    # Add nodes with their types as attributes
    if not G.has_node(source):
        G.add_node(source, node_type=source_type)
    if not G.has_node(target):
        G.add_node(target, node_type=target_type)
    
    # Add edge with relation type as attribute
    G.add_edge(source, target, relation=relation)

# 6. Prepare for visualization
plt.figure(figsize=(14, 10))

# Use a layout algorithm that provides good separation
pos = nx.spring_layout(G, k=0.5, seed=42)

# Draw nodes based on their types
for node_type in set(nx.get_node_attributes(G, 'node_type').values()):
    nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == node_type]
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=nodes,
        node_color=node_colors[node_type],
        node_shape=node_shapes[node_type],
        node_size=300 if node_type.startswith('places') else 500,
        alpha=0.8
    )

# Draw edges based on their types
for relation_type in edge_colors:
    edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('relation') == relation_type]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=edge_colors[relation_type],
        style=edge_styles[relation_type],
        width=1.5,
        alpha=0.7,
        arrowsize=15,
        connectionstyle='arc3,rad=0.1'  # Curved edges for better visualization
    )

# Draw node labels with smaller font for long names
node_labels = {}
for node in G.nodes():
    if len(str(node)) > 20:
        node_labels[node] = str(node)[:17] + '...'
    else:
        node_labels[node] = node

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

# Create a legend for node types
node_legend_elements = [
    mpatches.Patch(color=color, label=node_type.split(':')[1])
    for node_type, color in node_colors.items()
]

# Create a legend for edge types
edge_legend_elements = [
    plt.Line2D([0], [0], color=color, linestyle=edge_styles[relation], label=relation, lw=2)
    for relation, color in edge_colors.items()
]

# Place legends
combined_legend_elements = node_legend_elements + edge_legend_elements
combined_legend_titles = ["节点类型"] * len(node_legend_elements) + ["关系类型"] * len(edge_legend_elements)

# Create a legend with a handler map for different legend entry types
legend = plt.legend(
    handles=combined_legend_elements, 
    loc='upper left', 
    bbox_to_anchor=(1, 1)
)

# Add titles to the legend entries
for i, text in enumerate(legend.get_texts()):
    if i == 0:
        text.set_text("节点类型: " + text.get_text())
    elif i == len(node_legend_elements):
        text.set_text("关系类型: " + text.get_text())

plt.title('')
plt.axis('off')
plt.tight_layout()
plt.savefig('heterogeneous_network.png', dpi=300, bbox_inches='tight')
plt.gcf().canvas.mpl_connect('close_event', lambda evt: plt.close('all'))