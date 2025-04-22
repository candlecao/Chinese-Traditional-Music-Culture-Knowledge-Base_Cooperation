# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from collections import defaultdict
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Set up Chinese font support
import matplotlib.font_manager as fm
fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STHeiti', 'PingFang SC', 'Hiragino Sans GB']
chinese_font = 'sans-serif'
for font in fonts:
    try:
        if any([f for f in fm.fontManager.ttflist if font in f.name]):
            chinese_font = font
            print(f"Using Chinese font: {font}")
            break
    except:
        continue

plt.rcParams['font.family'] = chinese_font
plt.rcParams['axes.unicode_minus'] = False  # Ensure proper display of minus signs

# Define the CSV data
csv_data = '''乐种,县区,县区坐标（经纬度）,离东宝区的距离（km）
鄂北打调,宜城市,"POINT(112.25776 31.71976)","74.409"
鄂北打调-粗乐,宜城市,"POINT(112.25776 31.71976)","74.409"
鄂北打调-细乐,宜城市,"POINT(112.25776 31.71976)","74.409"
鄂北花鼓戏,远安县,"POINT(111.64132 31.06129)","53.3627"
江汉丝弦,当阳市,"POINT(111.78833 30.82108)","47.0211"
沮水巫音,远安县,"POINT(111.64132 31.06129)","53.3627"
梁山调,钟祥市,"POINT(112.58817 31.16797)","38.9657"
宜昌细乐,当阳市,"POINT(111.78833 30.82108)","47.0211"
远安花鼓戏,远安县,"POINT(111.64132 31.06129)","53.3627"
,东宝区,"POINT(112.20173 31.05192)","0.0"
,掇刀区,"POINT(112.20772 30.97307)","8.78137"
,沙洋县,"POINT(112.58854 30.70918)","53.0276"
'''

# Write to a temporary file and read with pandas
with open('temp_data.csv', 'w', encoding='utf-8') as f:
    f.write(csv_data)

# Read the CSV file
df = pd.read_csv('temp_data.csv')

# Process coordinates
def extract_coordinates(point_str):
    match = re.search(r'POINT\((\d+\.\d+) (\d+\.\d+)\)', point_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

# Fixed extract_distance function to handle the quoted values properly
def extract_distance(dist_str):
    try:
        str_value = str(dist_str)
        cleaned = str_value.replace('"', '').replace("'", "").strip()
        return float(cleaned)
    except Exception as e:
        print(f"Error parsing distance: {e} for value: '{dist_str}'")
        return 0.0

# Extract coordinates and distances
df['lon'], df['lat'] = zip(*df['县区坐标（经纬度）'].apply(extract_coordinates))
df['distance'] = df['离东宝区的距离（km）'].apply(extract_distance)

# Create a unique list of counties
counties = df[['县区', 'lon', 'lat', 'distance']].dropna(subset=['县区']).drop_duplicates().reset_index(drop=True)

# Create the graph
G: nx.DiGraph = nx.DiGraph()

# Initialize dongbao_coord with a default value
dongbao_coord = (0, 0)

# Add county nodes
for _, row in counties.iterrows():
    G.add_node(row['县区'], 
              pos=(row['lon'], row['lat']), 
              type='county',
              distance=row['distance'])
    if row['县区'] == '东宝区':
        dongbao_coord = (row['lon'], row['lat'])

# Group music types by county
music_types_by_county = defaultdict(list)
for _, row in df[df['乐种'].notna()].iterrows():
    music_types_by_county[row['县区']].append(row['乐种'])

# Add music type nodes
for music in df[df['乐种'].notna()]['乐种'].unique():
    G.add_node(music, type='music')

# Add distribution edges (music to county)
for _, row in df[df['乐种'].notna()].iterrows():
    G.add_edge(row['乐种'], row['县区'], type='distribution')

# Add distance edges from dongbao to other counties
for _, row in counties[counties['县区'] != '东宝区'].iterrows():
    G.add_edge('东宝区', row['县区'], 
              type='distance', 
              weight=row['distance'],
              distance=row['distance'])

# Function to calculate optimized positions for music nodes
def calculate_optimized_music_positions(G, county_positions):
    positions = {}
    county_to_music = defaultdict(list)
    
    # Group music by county
    for music in [n for n, d in G.nodes(data=True) if d.get('type') == 'music']:
        counties = list(G.neighbors(music))
        if counties:
            county_to_music[counties[0]].append(music)
    
    # Custom positioning for each county
    for county, music_list in county_to_music.items():
        county_pos = county_positions[county]
        num_music = len(music_list)
        
        # Different placement strategies based on county
        if county == '宜城市':
            # Place music nodes in a column to the left
            base_angle = 3.5  # Left side
            for i, music in enumerate(music_list):
                distance = 0.23 + i * 0.08
                pos_x = county_pos[0] + distance * np.cos(base_angle)
                pos_y = county_pos[1] + 0.08 * i * np.sin(base_angle)
                positions[music] = (pos_x, pos_y)
        
        elif county == '远安县':
            # Place in a semi-circle on the left side
            base_angle = 2.5
            for i, music in enumerate(music_list):
                angle = base_angle + (i - num_music/2) * 0.6
                distance = 0.2 + 0.02 * i
                pos_x = county_pos[0] + distance * np.cos(angle)
                pos_y = county_pos[1] + distance * np.sin(angle)
                positions[music] = (pos_x, pos_y)
                
        elif county == '当阳市':
            # Place below
            base_angle = 4.0
            for i, music in enumerate(music_list):
                angle = base_angle + i * 0.8
                distance = 0.25
                pos_x = county_pos[0] + distance * np.cos(angle)
                pos_y = county_pos[1] + distance * np.sin(angle)
                positions[music] = (pos_x, pos_y)
                
        elif county == '钟祥市':
            # Place to the right
            base_angle = 0.2
            distance = 0.22
            positions[music_list[0]] = (
                county_pos[0] + distance * np.cos(base_angle),
                county_pos[1] + distance * np.sin(base_angle)
            )
        
        else:
            # Generic placement for other counties
            base_angle = np.random.uniform(0, 2*np.pi)
            for i, music in enumerate(music_list):
                angle = base_angle + i * (2*np.pi / max(num_music, 1))
                distance = 0.25
                pos_x = county_pos[0] + distance * np.cos(angle)
                pos_y = county_pos[1] + distance * np.sin(angle)
                positions[music] = (pos_x, pos_y)
    
    return positions

# Get base positions for counties
pos = nx.get_node_attributes(G, 'pos')

# Calculate optimized positions for music nodes
music_pos = calculate_optimized_music_positions(G, pos)

# Update positions
pos.update(music_pos)

# Set node sizes based on type
node_sizes = {}
for node in G.nodes():
    if G.nodes[node]['type'] == 'county':
        node_sizes[node] = 600 if node == '东宝区' else 400
    else:
        node_sizes[node] = 300

# Set up the plot
plt.figure(figsize=(15, 12))
ax = plt.gca()

# Remove the outer rectangle border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Draw concentric circles around dongbao district
dongbao_lon, dongbao_lat = dongbao_coord
lat_to_km = 111.0  # km per degree latitude
max_distance = max([G.nodes[n].get('distance', 0) for n in G.nodes()])
circle_radii = np.arange(20, max_distance + 20, 20)

for radius_km in circle_radii:
    # Convert km to degrees
    radius_deg = radius_km / lat_to_km
    circle = plt.Circle((dongbao_lon, dongbao_lat), radius_deg, 
                       fill=False, linestyle='--', color='gray', alpha=0.5)
    ax.add_patch(circle)
    
    # Add a radius label at varied positions around the circle
    angle = np.pi/4 + (radius_km % 40) * 0.1  # Vary angle slightly
    label_x = dongbao_lon + radius_deg * np.cos(angle)
    label_y = dongbao_lat + radius_deg * np.sin(angle)
    ax.text(label_x, label_y, f"{radius_km}km", fontsize=12, ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Draw county nodes
county_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'county']
nx.draw_networkx_nodes(G, pos, 
                     nodelist=county_nodes,
                     node_size=[node_sizes[n] for n in county_nodes],
                     node_color='red',
                     alpha=0.8,
                     ax=ax)

# Draw music nodes
music_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'music']
nx.draw_networkx_nodes(G, pos, 
                     nodelist=music_nodes,
                     node_size=[node_sizes[n] for n in music_nodes],
                     node_color='green',
                     node_shape='^',
                     alpha=0.8,
                     ax=ax)

# Draw distribution edges with varied styles (solid green)
distribution_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'distribution']
for u, v in distribution_edges:
    # Calculate edge length to vary width
    u_pos, v_pos = pos[u], pos[v]
    edge_len = np.sqrt((u_pos[0] - v_pos[0])**2 + (u_pos[1] - v_pos[1])**2)
    
    # Longer edges slightly thinner for better aesthetics
    width = 2.2 - min(0.8, edge_len * 2)
    
    # Draw edge
    nx.draw_networkx_edges(G, pos,
                         edgelist=[(u, v)],
                         width=width,
                         alpha=0.7,
                         edge_color='green',
                         ax=ax)

# Draw distance edges (dashed blue with arrows)
distance_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'distance']
nx.draw_networkx_edges(G, pos,
                     edgelist=distance_edges,
                     width=1.5,
                     alpha=0.7,
                     edge_color='blue',
                     style='dashed',
                     arrows=True,
                     arrowstyle='-|>',
                     ax=ax)

# Add distance labels on edges
for u, v, data in G.edges(data=True):
    if data.get('type') == 'distance':
        # Get correct distance value
        distance_value = float(data.get('distance', 0))
        
        # Get positions
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Calculate edge length and orientation
        edge_len = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = np.arctan2(y2-y1, x2-x1)
        
        # Calculate offset based on edge characteristics
        offset_angle = angle + np.pi/2  # Perpendicular to edge
        offset_dist = 0.01 + (distance_value % 10) / 1000  # Small variation
        
        # Position label with offset to prevent uniform alignment
        frac = 0.5  # Mid-point
        mid_x = x1 + frac * (x2 - x1) + offset_dist * np.cos(offset_angle)
        mid_y = y1 + frac * (y2 - y1) + offset_dist * np.sin(offset_angle)
        
        # Add label with white background and black border
        ax.text(mid_x, mid_y, f"{distance_value:.1f}km", 
                fontsize=14, color='blue', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, pad=3),
                horizontalalignment='center', verticalalignment='center')

# Completely rewritten label manager that avoids connectors and prioritizes direct placement
class DirectLabelManager:
    def __init__(self, node_positions, node_sizes):
        self.positions = node_positions
        self.node_sizes = node_sizes
        self.occupied_areas = []
        
        # Define reserved space around nodes
        for node, (x, y) in node_positions.items():
            node_radius = np.sqrt(self.node_sizes[node] / np.pi) / 10000  # Convert to plot units
            self.occupied_areas.append({
                'center': (x, y),
                'radius': node_radius * 1.2,  # Add some buffer
                'type': 'node',
                'node': node
            })
    
    def rectangle_overlaps(self, x, y, width, height):
        """Check if a rectangle at (x,y) with given width/height overlaps any occupied area"""
        # Convert rectangle to corners
        half_w, half_h = width/2, height/2
        rect_points = [(x-half_w, y-half_h), (x+half_w, y-half_h), 
                     (x+half_w, y+half_h), (x-half_w, y+half_h)]
        
        # Check against circular occupied areas
        for area in self.occupied_areas:
            if area['type'] == 'node':
                # For node areas (circles), check if any corner is inside
                cx, cy = area['center']
                r = area['radius']
                for px, py in rect_points:
                    if (px-cx)**2 + (py-cy)**2 < r**2:
                        return True
            else:
                # For label areas (rectangles), check for overlap
                other_x, other_y = area['center']
                other_w, other_h = area['width'], area['height']
                
                # Check if rectangles overlap
                if (abs(x - other_x) < (width/2 + other_w/2)) and \
                   (abs(y - other_y) < (height/2 + other_h/2)):
                    return True
        
        return False
    
    def direct_adjacency_positions(self, node, label_text, font_size=20):
        """Get positions directly adjacent to the node, prioritizing clear directions"""
        pos = self.positions[node]
        node_size = self.node_sizes[node]
        
        # Calculate label dimensions
        label_len = len(label_text)
        label_width = 0.015 * label_len * font_size/10
        label_height = 0.03 * font_size/10
        
        # Calculate node radius in plot units
        node_radius = np.sqrt(node_size / np.pi) / 10000
        
        # Define 8 positions around the node (N, NE, E, SE, S, SW, W, NW)
        offsets = []
        
        # Custom offset suggestions for specific nodes to improve layout
        if node == '东宝区':
            # Center node - place label to the upper right
            offsets = [(node_radius * 1.2, node_radius * 1.2)]
        elif node == '沙洋县':
            # Try right side first
            offsets = [(node_radius * 1.2, 0), (node_radius * 0.8, node_radius * 0.8)]
        elif node == '掇刀区':
            # Try below first
            offsets = [(0, -node_radius * 1.2), (node_radius * 0.8, -node_radius * 0.8)]
        elif node == '钟祥市': 
            # Try upper left
            offsets = [(-node_radius * 0.8, node_radius * 0.8), (-node_radius * 1.2, 0)]
        elif node == '远安县':
            # Try above
            offsets = [(0, node_radius * 1.2), (node_radius * 0.8, node_radius * 0.8)]
        elif node == '当阳市':
            # Try lower left
            offsets = [(-node_radius * 0.8, -node_radius * 0.8), (0, -node_radius * 1.2)]
        elif node == '宜城市':
            # Try upper right
            offsets = [(node_radius * 0.8, node_radius * 0.8), (node_radius * 1.2, 0)]
        else:
            # For music nodes and other counties, use standard positions
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
            distance = node_radius * 1.1
            offsets = [(distance * np.cos(a), distance * np.sin(a)) for a in angles]
        
        # Test each position
        positions = []
        for dx, dy in offsets:
            label_x = pos[0] + dx
            label_y = pos[1] + dy
            
            # Check for overlaps
            if not self.rectangle_overlaps(label_x, label_y, label_width, label_height):
                positions.append({
                    'position': (label_x, label_y),
                    'font_size': font_size,
                    'offset': (dx, dy)
                })
        
        return positions
    
    def find_label_position(self, node):
        """Find best position for node label with no connectors if possible"""
        node_type = G.nodes[node]['type']
        base_font_size = 20 if node_type == 'county' else 18
        
        # Try direct adjacency first - most natural and no connectors
        direct_positions = self.direct_adjacency_positions(node, node, base_font_size)
        if direct_positions:
            best_pos = direct_positions[0]
            
            # Mark area as occupied
            label_len = len(node)
            label_width = 0.015 * label_len * best_pos['font_size']/10
            label_height = 0.03 * best_pos['font_size']/10
            
            self.occupied_areas.append({
                'center': best_pos['position'],
                'width': label_width,
                'height': label_height,
                'type': 'label',
                'node': node
            })
            
            return {
                'position': best_pos['position'],
                'font_size': best_pos['font_size'],
                'needs_connector': False
            }
        
        # Try with slightly smaller font
        direct_positions = self.direct_adjacency_positions(node, node, base_font_size - 2)
        if direct_positions:
            best_pos = direct_positions[0]
            
            # Mark area as occupied
            label_len = len(node)
            label_width = 0.015 * label_len * best_pos['font_size']/10
            label_height = 0.03 * best_pos['font_size']/10
            
            self.occupied_areas.append({
                'center': best_pos['position'],
                'width': label_width,
                'height': label_height,
                'type': 'label',
                'node': node
            })
            
            return {
                'position': best_pos['position'],
                'font_size': best_pos['font_size'],
                'needs_connector': False
            }
        
        # If we still can't place directly, use the more aggressive algorithm
        # at various distances
        node_pos = self.positions[node]
        node_radius = np.sqrt(self.node_sizes[node] / np.pi) / 10000
        outer_radius = node_radius * 2.5
        
        # Create test positions at various distances
        test_positions = []
        
        # Custom placement for specific nodes
        if node == '鄂北打调' or node == '鄂北打调-粗乐' or node == '鄂北打调-细乐':
            # For these nodes, we know they're near '宜城市'
            # Try specific placements to avoid crowding
            if node == '鄂北打调':
                angles = [2.8, 3.0, 3.2]
            elif node == '鄂北打调-粗乐':
                angles = [3.3, 3.5, 3.7]
            else:
                angles = [2.5, 2.7, 2.9]
                
            for angle in angles:
                for distance in [0.15, 0.18, 0.21]:
                    x = node_pos[0] + distance * np.cos(angle)
                    y = node_pos[1] + distance * np.sin(angle)
                    test_positions.append((x, y))
                    
        elif node == '远安花鼓戏' or node == '沮水巫音' or node == '鄂北花鼓戏':
            # For labels near '远安县'
            if node == '远安花鼓戏':
                angles = [3.8, 4.0, 4.2]
            elif node == '沮水巫音':
                angles = [4.3, 4.5, 4.7]
            else:
                angles = [3.5, 3.7, 3.9]
                
            for angle in angles:
                for distance in [0.16, 0.19, 0.22]:
                    x = node_pos[0] + distance * np.cos(angle)
                    y = node_pos[1] + distance * np.sin(angle)
                    test_positions.append((x, y))
        
        elif node == '江汉丝弦' or node == '宜昌细乐':
            # For labels near '当阳市'
            if node == '江汉丝弦':
                angles = [4.8, 5.0, 5.2]
            else:
                angles = [5.3, 5.5, 5.7]
                
            for angle in angles:
                for distance in [0.14, 0.17, 0.2]:
                    x = node_pos[0] + distance * np.cos(angle)
                    y = node_pos[1] + distance * np.sin(angle)
                    test_positions.append((x, y))
                    
        else:
            # For other nodes, try standard positions
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                for distance in [0.15, 0.2, 0.25]:
                    x = node_pos[0] + distance * np.cos(angle)
                    y = node_pos[1] + distance * np.sin(angle)
                    test_positions.append((x, y))
        
        # Test each position with smaller font
        smaller_font = base_font_size - 3  # Reduce font more aggressively
        label_len = len(node)
        label_width = 0.015 * label_len * smaller_font/10
        label_height = 0.03 * smaller_font/10
        
        for x, y in test_positions:
            if not self.rectangle_overlaps(x, y, label_width, label_height):
                # Mark area as occupied
                self.occupied_areas.append({
                    'center': (x, y),
                    'width': label_width,
                    'height': label_height,
                    'type': 'label',
                    'node': node
                })
                
                # We need a connector only if we're not adjacent to the node
                dx = x - node_pos[0]
                dy = y - node_pos[1]
                distance = np.sqrt(dx*dx + dy*dy)
                needs_connector = distance > outer_radius
                
                return {
                    'position': (x, y),
                    'font_size': smaller_font,
                    'needs_connector': needs_connector
                }
        
        # Last resort - use very small font and place it anywhere
        very_small_font = base_font_size - 4
        label_width = 0.015 * label_len * very_small_font/10
        label_height = 0.03 * very_small_font/10
        
        for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
            distance = 0.3  # Place further away
            x = node_pos[0] + distance * np.cos(angle)
            y = node_pos[1] + distance * np.sin(angle)
            
            if not self.rectangle_overlaps(x, y, label_width, label_height):
                # Mark area as occupied
                self.occupied_areas.append({
                    'center': (x, y),
                    'width': label_width,
                    'height': label_height,
                    'type': 'label',
                    'node': node
                })
                
                return {
                    'position': (x, y),
                    'font_size': very_small_font,
                    'needs_connector': True  # At this distance we need connector
                }
        
        # Absolute last resort - just return a position and hope for the best
        return {
            'position': (node_pos[0] + 0.4, node_pos[1] + 0.4),
            'font_size': very_small_font,
            'needs_connector': True
        }

# Initialize the new label manager
label_manager = DirectLabelManager(pos, node_sizes)

# Prepare for label drawing in optimal order
# Process counties first, then music nodes by importance
label_info = {}

# Counties first
county_nodes = sorted([n for n, d in G.nodes(data=True) if d.get('type') == 'county'],
                     key=lambda n: 0 if n == '东宝区' else 1)  # 东宝区 first

for node in county_nodes:
    label_info[node] = label_manager.find_label_position(node)

# Then music nodes by importance
music_nodes = sorted([n for n, d in G.nodes(data=True) if d.get('type') == 'music'],
                    key=lambda n: -len(list(G.neighbors(n))))

for node in music_nodes:
    label_info[node] = label_manager.find_label_position(node)

# Draw connector lines only where needed
for node, info in label_info.items():
    if info['needs_connector']:
        node_pos = pos[node]
        label_pos = info['position']
        
        # Draw a straight line with slightly transparent gray
        plt.plot([node_pos[0], label_pos[0]], [node_pos[1], label_pos[1]], 
                color='gray', linestyle='-', linewidth=0.7, alpha=0.5, zorder=5)

# Draw node labels
for node, info in label_info.items():
    label_pos = info['position']
    font_size = info['font_size']
    
    # Special handling for 东宝区 to make it more prominent
    if node == '东宝区':
        font_weight = 'bold'
    else:
        font_weight = 'normal'
    
    plt.text(label_pos[0], label_pos[1],
            node,
            fontsize=font_size,
            color='black',
            family=chinese_font,
            fontweight=font_weight,
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),
            zorder=10)

# Add legend
legend_elements = [
    Patch(facecolor='green', edgecolor='k', label='乐种', alpha=0.7),
    Patch(facecolor='red', edgecolor='k', label='县区', alpha=0.7),
    Line2D([0], [0], color='green', lw=2, label='乐种分布于县区', alpha=0.7),
    Line2D([0], [0], color='blue', linestyle='--', label='离东宝区距离', alpha=0.7),
    Line2D([0], [0], color='gray', linestyle='--', label='距离参考圈', alpha=0.7)
]

ax.legend(handles=legend_elements, loc='upper left', fontsize=14)

# Set title and labels with larger fonts
plt.title('湖北省乐种分布与县区距离关系图', fontsize=22)
plt.xlabel('经度', fontsize=16)
plt.ylabel('纬度', fontsize=16)

# Equal aspect ratio for proper geographic positioning
plt.axis('equal')

# Remove ticks and tick labels for a cleaner look
plt.xticks([])
plt.yticks([])

# Clean up temporary file
if os.path.exists('temp_data.csv'):
    os.remove('temp_data.csv')

# Save and show
plt.tight_layout()
plt.savefig('hubei_music_map.png', dpi=300, bbox_inches='tight')
plt.show()