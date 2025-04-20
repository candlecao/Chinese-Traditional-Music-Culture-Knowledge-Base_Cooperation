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
import time

# Start execution timer
start_time = time.time()

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
df = pd.read_csv('csvSimplified.csv')

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
# Normalize centrality values for node sizing (between 400 and 2000)
min_size, max_size = 400, 2000  # Increased sizes for better visibility
min_centrality = min(centrality.values())
max_centrality = max(centrality.values())
normalized_centrality = {
    node: min_size + (centrality[node] - min_centrality) * (max_size - min_size) / 
          (max_centrality - min_centrality) if max_centrality > min_centrality else min_size
    for node in centrality
}

# 7. Prepare for visualization with a larger figure
plt.figure(figsize=(22, 20))  # Large figure for better spacing

# 8. Use a sophisticated force-directed layout
# First, create an initial layout with strong repulsion to avoid node overlaps
print("Computing initial layout...")
pos_initial = nx.spring_layout(G, k=1.2, iterations=400, seed=42)

# 9. Use the Fruchterman-Reingold algorithm with optimized parameters for better node distribution
print("Refining layout with Fruchterman-Reingold algorithm...")
pos = nx.fruchterman_reingold_layout(G, k=0.3, iterations=700, pos=pos_initial, seed=42)

# 10. Now manually adjust nodes that are too close to each other
print("Fine-tuning node positions to reduce overlap...")
min_distance = 0.05  # Minimum distance between nodes

# Adjust any overlapping nodes
for i, node1 in enumerate(G.nodes()):
    for node2 in list(G.nodes())[i+1:]:
        pos1 = np.array(pos[node1])
        pos2 = np.array(pos[node2])
        distance = np.linalg.norm(pos1 - pos2)
        
        if distance < min_distance:
            # Calculate direction vector from node1 to node2
            direction = pos2 - pos1
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                # If nodes are exactly at the same position, use a random direction
                angle = np.random.random() * 2 * np.pi
                direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Move both nodes apart slightly
            adjustment = min_distance - distance
            pos[node1] = pos1 - direction * adjustment / 2
            pos[node2] = pos2 + direction * adjustment / 2

# 11. Draw nodes based on their types and centrality
print("Drawing nodes...")
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

# 12. Draw edges based on their types
print("Drawing edges...")
for relation_type in edge_colors:
    edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('relation') == relation_type]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=edge_colors[relation_type],
        style=edge_styles[relation_type],
        width=1.5,
        alpha=0.5,  # Reduced for better label visibility
        arrowsize=15,
        connectionstyle='arc3,rad=0.1'  # Curved edges to reduce crossings
    )

# 13. Label placement with multi-strategy approach
print("Planning label placement strategy...")
# Create a mapping of nodes to numeric IDs (for fallback)
node_to_id = {node: i+1 for i, node in enumerate(G.nodes())}
id_to_node = {i+1: node for i, node in enumerate(G.nodes())}

# Analyze node density and importance for prioritization
density_threshold = 0.12  # Adjusted to detect crowded areas
node_density = {}
node_neighbors = {}
for node, position in pos.items():
    # Count nodes within threshold distance
    nearby_nodes = 0
    neighbor_list = []
    for other_node, other_pos in pos.items():
        if node != other_node:
            dist = np.linalg.norm(np.array(position) - np.array(other_pos))
            if dist < density_threshold:
                nearby_nodes += 1
                neighbor_list.append((other_node, dist))
    node_density[node] = nearby_nodes
    node_neighbors[node] = sorted(neighbor_list, key=lambda x: float(x[1]))  # Sort by distance

# Sort nodes by importance (combination of centrality and node size)
importance_ranking = {
    node: centrality[node] * (1 - min(node_density[node] / 10, 0.8))  # Higher density reduces importance slightly
    for node in G.nodes()
}

# 14. Create an advanced label manager for sophisticated placement
class AdvancedLabelManager:
    def __init__(self, pos, node_sizes, canvas_bounds=(-1, 1, -1, 1)):
        self.pos = pos
        self.node_sizes = node_sizes  # Dictionary of node sizes
        self.grid_size = 0.02  # Small grid cell for precise collision detection
        self.occupied_cells = set()
        self.canvas_bounds = canvas_bounds  # (xmin, xmax, ymin, ymax)
        
        # Mark areas occupied by nodes
        for node, position in pos.items():
            node_radius = np.sqrt(node_sizes[node]/np.pi) / 100
            self._mark_area(position, node_radius)
    
    def _pos_to_grid(self, pos):
        # Convert a position to grid coordinates
        x_grid = int(pos[0] / self.grid_size)
        y_grid = int(pos[1] / self.grid_size)
        return (x_grid, y_grid)
    
    def _mark_area(self, pos, radius):
        # Mark an area as occupied (for nodes or labels)
        grid_pos = self._pos_to_grid(pos)
        grid_radius = int(radius / self.grid_size) + 1
        
        for i in range(-grid_radius, grid_radius+1):
            for j in range(-grid_radius, grid_radius+1):
                if i*i + j*j <= grid_radius*grid_radius:
                    self.occupied_cells.add((grid_pos[0]+i, grid_pos[1]+j))
    
    def _est_label_bounds(self, text, font_size):
        # Estimate the size taken by a label
        scale_factor = font_size / 10.0
        char_width = 0.012 * scale_factor
        width = len(text) * char_width
        height = 0.03 * scale_factor
        return width, height
    
    def try_direct_placement(self, node, text, font_size):
        """Try to place label directly at node with various offsets"""
        node_pos = np.array(self.pos[node])
        node_radius = np.sqrt(self.node_sizes[node]/np.pi) / 100
        width, height = self._est_label_bounds(text, font_size)
        
        # Try direct placement in different directions
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = np.cos(angle) * (node_radius + width/2 + 0.01)
            dy = np.sin(angle) * (node_radius + height/2 + 0.01)
            
            new_pos = (node_pos[0] + dx, node_pos[1] + dy)
            
            # Check if label would fit without overlapping anything
            if not self._check_overlap(new_pos, text, font_size):
                self._mark_label_area(new_pos, text, font_size)
                return new_pos, True
        
        return None, False  # Could not find direct placement
    
    def try_connector_placement(self, node, text, font_size, preferred_angles=None):
        """Try to place label with a connector line"""
        node_pos = np.array(self.pos[node])
        width, height = self._est_label_bounds(text, font_size)
        
        # Use provided angles or generate a set
        if preferred_angles is None:
            angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        else:
            angles = preferred_angles
            
        # Try different distances
        for distance in [0.15, 0.2, 0.25, 0.3, 0.4]:
            for angle in angles:
                dx = np.cos(angle) * distance
                dy = np.sin(angle) * distance
                
                new_pos = (node_pos[0] + dx, node_pos[1] + dy)
                
                # Check canvas bounds
                if not self._is_within_bounds(new_pos, width, height):
                    continue
                
                # Check if label would fit
                if not self._check_overlap(new_pos, text, font_size):
                    self._mark_label_area(new_pos, text, font_size)
                    return new_pos, True
        
        return None, False  # Could not find connector placement
    
    def _mark_label_area(self, pos, text, font_size):
        """Mark the area occupied by a label"""
        width, height = self._est_label_bounds(text, font_size)
        grid_pos = self._pos_to_grid(pos)
        
        # Calculate grid dimensions of the label
        grid_width = int(width / self.grid_size) + 1
        grid_height = int(height / self.grid_size) + 1
        
        # Mark all grid cells covered by the label
        for i in range(-grid_width//2, grid_width//2 + 1):
            for j in range(-grid_height//2, grid_height//2 + 1):
                self.occupied_cells.add((grid_pos[0]+i, grid_pos[1]+j))
    
    def _check_overlap(self, pos, text, font_size):
        """Check if a label would overlap with occupied areas"""
        width, height = self._est_label_bounds(text, font_size)
        grid_pos = self._pos_to_grid(pos)
        
        # Calculate grid dimensions of the label
        grid_width = int(width / self.grid_size) + 1
        grid_height = int(height / self.grid_size) + 1
        
        # Check all grid cells that would be covered by the label
        for i in range(-grid_width//2, grid_width//2 + 1):
            for j in range(-grid_height//2, grid_height//2 + 1):
                if (grid_pos[0]+i, grid_pos[1]+j) in self.occupied_cells:
                    return True  # Overlap detected
        return False  # No overlap
    
    def _is_within_bounds(self, pos, width, height):
        """Check if label stays within canvas bounds"""
        x_min, x_max, y_min, y_max = self.canvas_bounds
        half_width = width / 2
        half_height = height / 2
        
        return (pos[0] - half_width >= x_min and 
                pos[0] + half_width <= x_max and
                pos[1] - half_height >= y_min and
                pos[1] + half_height <= y_max)
    
    def try_numeric_id_placement(self, node, id_text, font_size=10):
        """Try to place a numeric ID near the node"""
        node_pos = np.array(self.pos[node])
        node_radius = np.sqrt(self.node_sizes[node]/np.pi) / 100
        
        # Try different positions very close to the node
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = np.cos(angle) * (node_radius + 0.03)
            dy = np.sin(angle) * (node_radius + 0.03)
            
            new_pos = (node_pos[0] + dx, node_pos[1] + dy)
            
            if not self._check_overlap(new_pos, id_text, font_size):
                self._mark_label_area(new_pos, id_text, font_size)
                return new_pos
        
        # If still can't place, try directly on top of the node
        if not self._check_overlap(node_pos, id_text, font_size):
            self._mark_label_area(node_pos, id_text, font_size)
            return node_pos
            
        return node_pos  # Last resort, place at node position anyway

# 15. Determine canvas bounds for label placement
x_values = [p[0] for p in pos.values()]
y_values = [p[1] for p in pos.values()]
canvas_bounds = (min(x_values)-0.2, max(x_values)+0.2, 
               min(y_values)-0.2, max(y_values)+0.2)

# 16. Create the advanced label manager
print("Setting up label manager...")
label_manager = AdvancedLabelManager(pos, normalized_centrality, canvas_bounds)

# 17. Process nodes for labeling based on importance
print("Processing node labels...")
sorted_nodes = sorted(list(G.nodes()), key=lambda n: importance_ranking[n], reverse=True)

# Prepare data structures for label info
label_positions = {}
node_labels = {}
font_sizes = {}
connector_lines = []
id_label_nodes = []

# Define font size categories
def get_base_font_size(node):
    """Determine base font size based on node importance and density"""
    if importance_ranking[node] > 0.8 * max(importance_ranking.values()):
        return 16  # Very important nodes
    elif importance_ranking[node] > 0.5 * max(importance_ranking.values()):
        return 14  # Important nodes
    elif importance_ranking[node] > 0.3 * max(importance_ranking.values()):
        return 12  # Moderately important
    elif importance_ranking[node] > 0.1 * max(importance_ranking.values()):
        return 11  # Less important
    else:
        return 10  # Least important

# Process each node according to the prioritized strategies
for node in sorted_nodes:
    node_pos = np.array(pos[node])
    text = str(node)
    base_font_size = get_base_font_size(node)
    
    # Strategy 1: Try direct placement with base font size
    direct_pos, success = label_manager.try_direct_placement(node, text, base_font_size)
    if success:
        label_positions[node] = direct_pos
        node_labels[node] = text
        font_sizes[node] = base_font_size
        continue
    
    # Strategy 2: Try with connector using base font size
    # Calculate preferred angles (away from neighbors)
    preferred_angles = []
    if node in node_neighbors and node_neighbors[node]:
        # Calculate angles to avoid (where neighbors are)
        avoid_angles = []
        for neighbor, _ in node_neighbors[node]:
            if neighbor in pos:
                neighbor_dir = np.array(pos[neighbor]) - node_pos
                if np.linalg.norm(neighbor_dir) > 0:
                    avoid_angle = np.arctan2(neighbor_dir[1], neighbor_dir[0])
                    avoid_angles.append(avoid_angle)
        
        # Generate angles that are far from neighbors
        for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
            if all(min(abs(angle - a), abs(angle - a + 2*np.pi), abs(angle - a - 2*np.pi)) > 0.3 for a in avoid_angles):
                preferred_angles.append(angle)
    
    # If no good angles found, use all angles
    if not preferred_angles:
        preferred_angles = []
    
    connector_pos, success = label_manager.try_connector_placement(node, text, base_font_size, preferred_angles)
    if success:
        label_positions[node] = connector_pos
        node_labels[node] = text
        font_sizes[node] = base_font_size
        connector_lines.append((pos[node], connector_pos))
        continue
    
    # Strategy 3: Try direct placement with reduced font size
    reduced_font_size = max(base_font_size - 3, 9)
    direct_pos, success = label_manager.try_direct_placement(node, text, reduced_font_size)
    if success:
        label_positions[node] = direct_pos
        node_labels[node] = text
        font_sizes[node] = reduced_font_size
        continue
    
    # Strategy 4: Try connector with reduced font size
    connector_pos, success = label_manager.try_connector_placement(node, text, reduced_font_size, preferred_angles)
    if success:
        label_positions[node] = connector_pos
        node_labels[node] = text
        font_sizes[node] = reduced_font_size
        connector_lines.append((pos[node], connector_pos))
        continue
    
    # Strategy 5: Last resort - numeric ID
    # Only do this for less important nodes
    if importance_ranking[node] < 0.4 * max(importance_ranking.values()):
        id_text = str(node_to_id[node])
        id_pos = label_manager.try_numeric_id_placement(node, id_text)
        label_positions[node] = id_pos
        node_labels[node] = id_text
        font_sizes[node] = 10  # Fixed size for IDs
        id_label_nodes.append(node)
        continue
    
    # Absolute last resort for important nodes - try far placement
    min_font = 8
    far_placed = False
    # Remove the unnecessary variable declaration:
    # far_distance: float = 0.0  # Explicitly type far_distance as a float
    
    # Try placing with increasing distances
    for far_distance in [0.5, 0.6, 0.7]:  # type: float
        if far_placed:
            break
        for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):  # Try more angles
            far_pos = (node_pos[0] + np.cos(angle) * distance, 
                      node_pos[1] + np.sin(angle) * distance)
            
            if not label_manager._check_overlap(far_pos, text, min_font):
                label_positions[node] = far_pos
                node_labels[node] = text
                font_sizes[node] = min_font
                connector_lines.append((pos[node], far_pos))
                far_placed = True
                break
    
    # If still no placement found, use numeric ID as absolute last resort
    if node not in label_positions:
        id_text = str(node_to_id[node])
        id_pos = label_manager.try_numeric_id_placement(node, id_text)
        label_positions[node] = id_pos
        node_labels[node] = id_text
        font_sizes[node] = 10
        id_label_nodes.append(node)

# 18. Draw connector lines
print("Drawing connector lines...")
for start, end in connector_lines:
    plt.plot(
        [start[0], end[0]], 
        [start[1], end[1]], 
        color='gray', linestyle='-', linewidth=1.0, alpha=0.6
    )

# 19. Draw actual labels
print("Drawing node labels...")
for node in G.nodes():
    if node not in label_positions:
        continue
    
    position = label_positions[node]
    label_text = node_labels[node]
    font_size = font_sizes[node]
    
    # Different styling for IDs vs. regular labels
    if node in id_label_nodes:
        # Numeric ID style
        plt.text(
            position[0], position[1],
            label_text,
            fontsize=font_size,
            fontweight='bold',
            color='black',
            bbox=dict(facecolor='#FFFFCC', edgecolor='black', alpha=0.9, pad=0.3, boxstyle='round'),
            family=chinese_font,
            horizontalalignment='center',
            verticalalignment='center',
            zorder=100
        )
    else:
        # Regular label style
        plt.text(
            position[0], position[1],
            label_text,
            fontsize=font_size,
            fontweight='bold',
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.3),
            family=chinese_font,
            horizontalalignment='center',
            verticalalignment='center',
            zorder=100
        )

# 20. Create legends
print("Creating legend...")
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

# 21. Save main visualization
print("Saving main visualization...")
plt.axis('off')
plt.tight_layout()
plt.savefig('heterogeneous_network.png', dpi=300, bbox_inches='tight')

# 22. Create a separate figure for the node ID mapping if needed
if id_label_nodes:
    print("Creating ID mapping legend...")
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

# Report execution time
end_time = time.time()
print(f"Visualization complete! Total execution time: {end_time - start_time:.2f} seconds")