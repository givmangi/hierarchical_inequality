import networkx as nx
import seaborn as sns
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 420
random.seed(SEED)
np.random.seed(SEED)

def generate_societies(n=2000, minority_pct=0.2):
    """
    n: Number of agents
    minority_pct: Percentage of agents in Group B (Marginalized)
    """
    
    hierarchy_G = nx.barabasi_albert_graph(n, m=2, seed = SEED) #barabasi net
    solidarity_G = nx.watts_strogatz_graph(n, k=4, p=0.1, seed = SEED) #solidarity net

    agents = list(range(n))
    random.shuffle(agents)
    num_minority = int(n * minority_pct)
    group_b = set(agents[:num_minority])
    

    for node in hierarchy_G.nodes():
        hierarchy_G.nodes[node]['group'] = 'B' if node in group_b else 'A'
        hierarchy_G.nodes[node]['wealth'] = 7.0
        if hierarchy_G == hierarchy_G:
            top_hubs = sorted(hierarchy_G.degree, key=lambda x: x[1], reverse=True)[:10]
            for hub_id, degree in top_hubs:
                hierarchy_G.nodes[hub_id]['group'] = 'A'
                if hierarchy_G == hierarchy_G:
                    hierarchy_G.nodes[hub_id]['wealth'] = 20.0

    for node in solidarity_G.nodes():
        solidarity_G.nodes[node]['group'] = 'B' if node in group_b else 'A'
        solidarity_G.nodes[node]['wealth'] = 10.0

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
            colors.append('blue')
        else:
            colors.append('red') 
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

def simulate_biased_step(G, homophily_alpha=0.1):
    """
    G: The network (Hierarchy or Solidarity)
    homophily_alpha: The 'Bias' level (0 = random, 1 = maximum bias)
    """
    potential_donors = [n for n, attr in G.nodes(data=True) if attr['wealth'] >= 10] 
    #able to exchanhge only when wealth gt 10 
    
    for donor in potential_donors:
        neighbors = list(G.neighbors(donor))
        if not neighbors: 
            continue
        donor_group = G.nodes[donor]['group']
        probs = []
        for nbr in neighbors:
            nbr_group = G.nodes[nbr]['group']
            if nbr_group == donor_group:
                probs.append(1 + homophily_alpha)
            else:
                probs.append(1 - homophily_alpha) 
        probs = np.array(probs) / sum(probs)
        recipient = np.random.choice(neighbors, p=probs)
        G.nodes[donor]['wealth'] -= 1.0
        G.nodes[recipient]['wealth'] += 1.1 

def run_experiment(h_net, s_net, steps=100, alpha=0.2):
    results = []
    for t in range(steps):
        simulate_biased_step(h_net, homophily_alpha=alpha)
        simulate_biased_step(s_net, homophily_alpha=alpha)
        # record avg wealth for minority in each model
        h_wealth_B = np.mean([d['wealth'] for n, d in h_net.nodes(data=True) if d['group'] == 'B'])
        h_wealth_A= np.mean([d['wealth'] for n, d in h_net.nodes(data=True) if d['group'] == 'A'])
        s_wealth_B = np.mean([d['wealth'] for n, d in s_net.nodes(data=True) if d['group'] == 'B'])
        s_wealth_A= np.mean([d['wealth'] for n, d in s_net.nodes(data=True) if d['group'] == 'A'])
        results.append({
            'step': t,
            'Hierarchy_GroupB': h_wealth_B,
            'Hierarchy_GroupA': h_wealth_A,
            'Solidarity_GroupB': s_wealth_B,
            'Solidarity_GroupA': s_wealth_A
        })
    return pd.DataFrame(results)

def load_nets(hierarchy_path, solidarity_path):
    try:
        h_net = nx.read_graphml(hierarchy_path)
        s_net = nx.read_graphml(solidarity_path)
    except Exception as e:
        print(f"Error loading graphs: {e}\n generating new ones instead.")
        h_net, s_net = generate_societies()
        nx.write_graphml(h_net, "hierarchy_network.graphml")
        nx.write_graphml(s_net, "solidarity_network.graphml")
        draw_nets(h_net, s_net)           
    for G in [h_net, s_net]:
        for node in G.nodes():
            G.nodes[node]['wealth'] = float(G.nodes[node]['wealth'])
    return h_net, s_net

def plot_wealth_gap(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='Hierarchy_GroupB', label='Hierarchy (Scale-Free)', color='blue', linewidth=2)
    sns.lineplot(data=df, x='step', y='Solidarity_GroupB', label='Solidarity (Small-World)', color='green', linewidth=2)
    plt.title("Minority Wealth Growth: Hierarchy vs. Solidarity Models", fontsize=16)
    plt.xlabel("Simulation Steps (Time)", fontsize=12)
    plt.ylabel("Average Wealth of Group B", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)        
    plt.show()

def main():
    h_net, s_net = load_nets("hierarchy_network.graphml", "solidarity_network.graphml")
    print(f"Hierarchy Nodes: {h_net.number_of_nodes()}, Edges: {h_net.number_of_edges()}")
    print(f"Solidarity Nodes: {s_net.number_of_nodes()}, Edges: {s_net.number_of_edges()}")
    try:
        history = pd.read_csv("simulation_history.csv")
        print("Loaded existing simulation history.")
    except FileNotFoundError:
        history = run_experiment(h_net, s_net, steps=2000, alpha=0.3)
        history.to_csv("simulation_history.csv", index=False)
    print(history.tail())
    plot_wealth_gap(history)

if __name__ == "__main__":
    main()