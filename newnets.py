import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt

def generate_societies(n=2000, minority_pct=0.2):
    """
    n: Number of agents
    minority_pct: Percentage of agents in Group B (Marginalized)
    """
    
    hierarchy_G = nx.barabasi_albert_graph(n, m=2) #barabasi net
    solidarity_G = nx.watts_strogatz_graph(n, k=4, p=0.1) #solidarity net

    agents = list(range(n))
    random.shuffle(agents)
    num_minority = int(n * minority_pct)
    group_b = set(agents[:num_minority])
    
    for G in [hierarchy_G, solidarity_G]:
        for node in G.nodes():
            G.nodes[node]['group'] = 'B' if node in group_b else 'A'
            G.nodes[node]['wealth'] = 10.0
            if G == hierarchy_G:
                top_hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
                for hub_id, degree in top_hubs:
                    G.nodes[hub_id]['group'] = 'A'

    return hierarchy_G, solidarity_G

def get_plotting_attributes(G):
    """
    Helper function to generate color lists and size lists based on
    node attributes (group and degree).
    """
    colors = []
    sizes = []
    for node in G.nodes():
        if G.nodes[node]['group'] == 'A':
            colors.append('#1f77b4') # standard matplotlib blue
        else:
            colors.append('#d62728') # standard matplotlib red
        deg = G.degree[node]
        sizes.append((deg * 30) + 50)        
    return colors, sizes

def draw_nets(h_net, s_net):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    ax1.set_title("The Hierarchy\n(Scale-Free / Preferential Attachment)", fontsize=16, fontweight='bold')

    pos_h = nx.spring_layout(h_net, k=0.15, iterations=50, seed=42)
    h_colors, h_sizes = get_plotting_attributes(h_net)
    nx.draw_networkx_edges(h_net, pos_h, ax=ax1, alpha=0.1) 
    nx.draw_networkx_nodes(h_net, pos_h, ax=ax1, 
                        node_color=h_colors, 
                        node_size=h_sizes, 
                        edgecolors='white', linewidths=1)
    ax1.axis('off') 

    #___plot2____
    ax2.set_title("The Solidarity Network\n(Small-World / Decentralized)", fontsize=16, fontweight='bold')

    pos_s = nx.spring_layout(s_net, k=0.2, iterations=50, seed=42) 
    s_colors, s_sizes = get_plotting_attributes(s_net)
    nx.draw_networkx_edges(s_net, pos_s, ax=ax2, alpha=0.1)
    nx.draw_networkx_nodes(s_net, pos_s, ax=ax2, 
                        node_color=s_colors, 
                        node_size=s_sizes, 
                        edgecolors='white', linewidths=1)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    h_net, s_net = generate_societies()
    draw_nets(h_net, s_net)
    print(f"Hierarchy Nodes: {h_net.number_of_nodes()}, Edges: {h_net.number_of_edges()}")
    print(f"Solidarity Nodes: {s_net.number_of_nodes()}, Edges: {s_net.number_of_edges()}")

if __name__ == "__main__":
    main()