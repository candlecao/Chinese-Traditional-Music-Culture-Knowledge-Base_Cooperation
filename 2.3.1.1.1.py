import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures
# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
import matplotlib.font_manager as fm
import os

# Check available fonts that support Chinese
def get_font_support_chinese():
    # First try system fonts that are known to support Chinese
    system_fonts = ['Arial Unicode MS', 'STHeiti', 'SimHei', 
                   'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 
                   'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN']
    
    # Check if any of these fonts are available
    for font in system_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font))
            if font_path and 'fallback' not in font_path.lower():
                return font
        except:
            continue
    
    # If no system font is found, return a safe default
    return 'sans-serif'

# Get a good Chinese font
chinese_font = get_font_support_chinese()

# Set font configuration for Chinese characters globally
plt.rcParams['font.sans-serif'] = [chinese_font, 'Arial Unicode MS', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# 6. Calculate node centrality for node sizing
# Use degree centrality as a measure of importance
centrality = nx.degree_centrality(G)
# Normalize centrality values for node sizing (between 300 and 1200)
min_size, max_size = 300, 1200  # Increased max size for better visibility
min_centrality = min(centrality.values())
max_centrality = max(centrality.values())
normalized_centrality = {
    node: min_size + (centrality[node] - min_centrality) * (max_size - min_size) / 
          (max_centrality - min_centrality) if max_centrality > min_centrality else min_size
    for node in centrality
}

# 7. Prepare for visualization
plt.figure(figsize=(24, 22))  # Very large figure size to allow for label spacing

# Use an optimized force-directed layout with parameters to minimize edge crossings
# Higher iterations and k value for better node separation
pos = nx.spring_layout(G, k=1.5, iterations=500, seed=42)

# 8. Draw nodes based on their types and centrality
for node_type in set(nx.get_node_attributes(G, 'node_type').values()):
    nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == node_type]
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=nodes,
        node_color=node_colors[node_type],
        node_shape=node_shapes[node_type],
        node_size=[normalized_centrality[node] for node in nodes],
        alpha=0.8
    )

# 9. Draw edges based on their types
for relation_type in edge_colors:
    edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('relation') == relation_type]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=edge_colors[relation_type],
        style=edge_styles[relation_type],
        width=1.5,
        alpha=0.4,  # Further reduced alpha for better label visibility
        arrowsize=15,
        connectionstyle='arc3,rad=0.1'  # Curved edges for better visualization
    )

# 10. Handle node labels with multiple strategies to prevent overlap
# Create a mapping of nodes to numeric IDs (only for fallback)
node_to_id = {node: i+1 for i, node in enumerate(G.nodes())}
id_to_node = {i+1: node for i, node in enumerate(G.nodes())}

# 10.1 Group nodes by proximity to detect overlapping regions
proximity_threshold = 0.2  # Increased threshold for better clustering detection
position_groups = defaultdict(list)
for node, position in pos.items():
    group_key = (round(position[0] / proximity_threshold), 
                round(position[1] / proximity_threshold))
    position_groups[group_key].append(node)

# 10.2 Calculate node density - how many nodes are in very close proximity
node_density = {}
for node, position in pos.items():
    count = 0
    for other_node, other_pos in pos.items():
        if node != other_node:
            dist = np.linalg.norm(np.array(position) - np.array(other_pos))
            if dist < proximity_threshold:
                count += 1
    node_density[node] = count

# 10.3 Classify nodes based on display strategy
# Strategy 1: Direct label display (isolated or low density nodes)
# Strategy 2: Offset labels with connectors (medium density)
# Strategy 3: Numeric IDs (very high density areas)

# Sort nodes by density and centrality for prioritization
sorted_by_density = sorted(list(G.nodes()), key=lambda n: (node_density[n], -centrality[n]))

# Calculate the proportion of nodes for each strategy
total_nodes = len(list(G.nodes()))
direct_label_cutoff = int(total_nodes * 0.5)  # 50% get direct labels
connector_label_cutoff = int(total_nodes * 0.85)  # 35% get connector labels

direct_label_nodes = sorted_by_density[:direct_label_cutoff]
connector_label_nodes = sorted_by_density[direct_label_cutoff:connector_label_cutoff]
id_label_nodes = sorted_by_density[connector_label_cutoff:]

# Ensure very high centrality nodes always get proper labels
top_central_nodes = sorted(list(G.nodes()), key=lambda n: centrality[n], reverse=True)[:int(total_nodes * 0.1)]
for node in top_central_nodes:
    if node in id_label_nodes:
        id_label_nodes.remove(node)
        connector_label_nodes.append(node)

# 10.4 Calculate label positions with improved anti-overlap algorithm
label_pos = {}
node_labels = {}
id_needed = False  # Track if we need to create an ID mapping file

# Process direct label nodes (strategy 1)
for node in direct_label_nodes:
    # Place label directly at node position with small offset
    offset = 0.01 + (normalized_centrality[node] / max_size * 0.01)
    label_pos[node] = (pos[node][0], pos[node][1] + offset)
    node_labels[node] = node

# Process connector label nodes (strategy 2) with improved placement
connector_lines = []  # Store line coordinates for later drawing

# Group connector nodes by sectors to distribute them evenly
num_sectors = 12
sector_angle = 2 * np.pi / num_sectors
sector_counts = [0] * num_sectors
sector_max_count = 5  # Maximum number of labels in each sector

# First pass: assign each node to a sector based on its position
node_sectors = {}
for node in connector_label_nodes:
    # Calculate angle between node and center of graph
    node_pos = np.array(pos[node])
    angle = np.arctan2(node_pos[1], node_pos[0])
    if angle < 0:
        angle += 2 * np.pi
    
    # Calculate sector index
    sector_idx = int(angle / sector_angle)
    node_sectors[node] = sector_idx
    sector_counts[sector_idx] += 1

# Second pass: adjust sectors with too many nodes
for node in sorted(connector_label_nodes, key=lambda n: centrality[n], reverse=True):
    sector_idx = node_sectors[node]
    if sector_counts[sector_idx] > sector_max_count:
        # Try to find a less crowded nearby sector
        for offset in [1, -1, 2, -2]:
            new_sector = (sector_idx + offset) % num_sectors
            if sector_counts[new_sector] < sector_max_count:
                sector_counts[sector_idx] -= 1
                sector_counts[new_sector] += 1
                node_sectors[node] = new_sector
                break

# Third pass: position labels within their sectors
for node in connector_label_nodes:
    sector_idx = node_sectors[node]
    base_angle = sector_idx * sector_angle + sector_angle/2
    
    # Calculate appropriate offset distance based on node centrality and density
    base_distance = 0.1 + 0.2 * (node_density[node] / max(node_density.values()))
    
    # Add some variation within the sector to prevent exact overlaps
    variation = (sector_counts[sector_idx] > 1) 
    if variation:
        angle_offset = (np.random.random() - 0.5) * sector_angle * 0.7
    else:
        angle_offset = 0
    
    # Calculate final label position
    final_angle = base_angle + angle_offset
    offset_x = np.cos(final_angle) * base_distance
    offset_y = np.sin(final_angle) * base_distance
    
    label_pos[node] = (
        pos[node][0] + offset_x,
        pos[node][1] + offset_y
    )
    node_labels[node] = node
    
    # Store connector line coordinates
    connector_lines.append((pos[node], label_pos[node]))

# Process ID label nodes (strategy 3) - only if there are any
if id_label_nodes:
    id_needed = True
    for node in id_label_nodes:
        label_pos[node] = pos[node]
        node_labels[node] = str(node_to_id[node])

# 10.5 Draw connector lines for offset labels
for start, end in connector_lines:
    plt.plot(
        [start[0], end[0]], 
        [start[1], end[1]], 
        color='gray', linestyle='-', linewidth=0.7, alpha=0.5
    )

# 10.6 Draw all node labels with background boxes for better readability
nx.draw_networkx_labels(
    G, label_pos, labels=node_labels, 
    font_size=9, font_weight='bold', 
    font_color='black',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3),
    font_family=chinese_font
)

# 11. Create legends
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

# Place main legends
plt.legend(
    handles=node_legend_elements + edge_legend_elements,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="图例 (Legend)",
    fontsize=12,
    prop={'family': chinese_font}
)

# Save main visualization
plt.axis('off')
plt.tight_layout()
plt.savefig('heterogeneous_network.png', dpi=300, bbox_inches='tight')

# Create a separate figure for the node ID mapping only if we used IDs
if id_needed:
    fig, ax = plt.subplots(figsize=(14, max(10, len(id_label_nodes) // 3)))
    plt.axis('off')
    
    # Create the mapping text
    mapping_text = "节点ID映射 (Node ID Mapping):\n\n"
    
    # Group the mapping by node type for better readability
    type_to_nodes = defaultdict(list)
    for node in id_label_nodes:
        node_type = G.nodes[node].get('node_type', 'Unknown')
        type_to_nodes[node_type].append(node)
    
    for node_type, nodes in type_to_nodes.items():
        # Format the node type more nicely
        type_label = node_type.split(':')[1] if ':' in node_type else node_type
        mapping_text += f"{type_label}:\n"
        for node in sorted(nodes, key=lambda n: node_to_id[n]):
            mapping_text += f"  {node_to_id[node]}: {node}\n"
        mapping_text += "\n"
    
    # Display text directly with explicit font family
    ax.text(0.05, 0.95, mapping_text,
            family=chinese_font,
            fontsize=12,
            verticalalignment='top', 
            horizontalalignment='left')
    
    plt.tight_layout()
    plt.savefig('node_id_mapping.png', dpi=300, bbox_inches='tight')

plt.close('all')