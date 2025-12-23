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

    top_hubs = sorted(hierarchy_G.degree, key=lambda x: x[1], reverse=True)[:100]
    for hub_id, degree in top_hubs:
        hierarchy_G.nodes[hub_id]['group'] = 'A'
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
    potential_donors = [n for n, attr in G.nodes(data=True) if attr['wealth'] >= 8.5] 
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

def get_gini(wealth_list):
    array = np.sort(np.array(wealth_list))
    n = len(array)
    if n == 0 or np.sum(wealth_list) == 0:
        return 0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * wealth_list)) / (n * np.sum(wealth_list))

def run_experiment(h_net, s_net, steps=100, alpha=0.1):
    h_net_unbiased = h_net.copy()
    s_net_unbiased = s_net.copy()
    results = []
    for i in range(steps):
        simulate_biased_step(h_net, homophily_alpha=alpha)
        simulate_biased_step(s_net, homophily_alpha=alpha)
        h_wealth_B_biased = np.mean([d['wealth'] for n, d in h_net.nodes(data=True) if d['group'] == 'B'])
        h_wealth_A_biased = np.mean([d['wealth'] for n, d in h_net.nodes(data=True) if d['group'] == 'A'])
        s_wealth_B_biased = np.mean([d['wealth'] for n, d in s_net.nodes(data=True) if d['group'] == 'B'])
        s_wealth_A_biased = np.mean([d['wealth'] for n, d in s_net.nodes(data=True) if d['group'] == 'A'])
        h_gini_biased = get_gini([d['wealth'] for n, d in h_net.nodes(data=True)])
        s_gini_biased = get_gini([d['wealth'] for n, d in s_net.nodes(data=True)])

        simulate_biased_step(h_net_unbiased, homophily_alpha=0.0)
        simulate_biased_step(s_net_unbiased, homophily_alpha=0.0)
        h_wealth_B_unbiased = np.mean([d['wealth'] for n, d in h_net_unbiased.nodes(data=True) if d['group'] == 'B'])
        h_wealth_A_unbiased = np.mean([d['wealth'] for n, d in h_net_unbiased.nodes(data=True) if d['group'] == 'A'])
        s_wealth_B_unbiased = np.mean([d['wealth'] for n, d in s_net_unbiased.nodes(data=True) if d['group'] == 'B'])
        s_wealth_A_unbiased = np.mean([d['wealth'] for n, d in s_net_unbiased.nodes(data=True) if d['group'] == 'A'])
        h_gini_unbiased = get_gini([d['wealth'] for n, d in h_net_unbiased.nodes(data=True)])
        s_gini_unbiased = get_gini([d['wealth'] for n, d in s_net_unbiased.nodes(data=True)]) 
        results.append({
            'step': i,
            #biased step
            'H_B_biased': h_wealth_B_biased,
            'H_A_biased': h_wealth_A_biased,
            'S_B_biased': s_wealth_B_biased,
            'S_A_biased': s_wealth_A_biased,
            'H_Gini_biased': h_gini_biased,
            'S_Gini_biased': s_gini_biased,
            #unbiased steps
            'H_B_unbiased': h_wealth_B_unbiased,
            'H_A_unbiased': h_wealth_A_unbiased,
            'S_B_unbiased': s_wealth_B_unbiased,
            'S_A_unbiased': s_wealth_A_unbiased,
            'H_Gini_unbiased': h_gini_unbiased,
            'S_Gini_unbiased': s_gini_unbiased,
        })
    return pd.DataFrame(results), h_net, s_net

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

def plot_results(df):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    sns.lineplot(data=df, x='step', y='H_B_biased', label='Hierarchy (Biased)', color='#1f77b4', linestyle='-', ax=ax1)
    sns.lineplot(data=df, x='step', y='H_B_unbiased', label='Hierarchy (Unbiased)', color='#1f77b4', linestyle='--', ax=ax1)

    sns.lineplot(data=df, x='step', y='S_B_biased', label='Solidarity (Biased)', color='#2ca02c', linestyle='-', ax=ax1)
    sns.lineplot(data=df, x='step', y='S_B_unbiased', label='Solidarity (Unbiased)', color='#2ca02c', linestyle='--', ax=ax1)

    ax1.set_title("Minority Group B Average Wealth: Topological & Behavioral Impact", fontsize=15, fontweight='bold')
    ax1.set_ylabel("Average Wealth", fontsize=12)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))


    sns.lineplot(data=df, x='step', y='H_Gini_biased', label='Hierarchy (Biased)', color='#1f77b4', linestyle='-', ax=ax2)
    sns.lineplot(data=df, x='step', y='H_Gini_unbiased', label='Hierarchy (Unbiased)', color='#1f77b4', linestyle='--', ax=ax2)

    sns.lineplot(data=df, x='step', y='S_Gini_biased', label='Solidarity (Biased)', color='#2ca02c', linestyle='-', ax=ax2)
    sns.lineplot(data=df, x='step', y='S_Gini_unbiased', label='Solidarity (Unbiased)', color='#2ca02c', linestyle='--', ax=ax2)

    ax2.set_title("Systemic Inequality (Gini Coefficient) Over Time", fontsize=15, fontweight='bold')
    ax2.set_ylabel("Gini Index (0=Equal, 1=Concentrated)", fontsize=12)
    ax2.set_xlabel("Simulation Steps", fontsize=12)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

def plot_wealth_gap(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='step', y='Hierarchy_GroupB', label='Hierarchy (Scale-Free)', color='blue', linewidth=2)
    sns.lineplot(data=df, x='step', y='Solidarity_GroupB', label='Solidarity (Small-World)', color='green', linewidth=2)
    plt.title("Minority Wealth Growth: Hierarchy vs. Solidarity Models", fontsize=16)
    plt.xlabel("Simulation Steps (Time)", fontsize=12)
    plt.ylabel("Average Wealth of Group B", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)        
    plt.savefig("img/wealth_gap.png", dpi=300)

def plot_lorenz_curve(h_net, s_net):
    def get_lorenz(G):
        wealths = np.sort([d['wealth'] for n, d in G.nodes(data=True)])
        cum_wealth = np.cumsum(wealths) / np.sum(wealths)
        return np.insert(cum_wealth, 0, 0)

    plt.figure(figsize=(8, 8))
    
    plt.plot(np.linspace(0, 1, len(h_net)+1), get_lorenz(h_net), label='Hierarchy', color='#1f77b4', lw=3)
    plt.plot(np.linspace(0, 1, len(s_net)+1), get_lorenz(s_net), label='Solidarity', color='#2ca02c', lw=3)
    
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Perfect Equality')
    plt.title("Lorenz Curve: Wealth Distribution at End of Simulation", fontsize=14)
    plt.xlabel("Fraction of Population (Poorest to Richest)")
    plt.ylabel("Fraction of Total Wealth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("img/lorenz_curve.png", dpi=300)
    plt.show()

def main():
    h_net, s_net = load_nets("hierarchy_network.graphml", "solidarity_network.graphml")
    print(f"Hierarchy Nodes: {h_net.number_of_nodes()}, Edges: {h_net.number_of_edges()}")
    print(f"Solidarity Nodes: {s_net.number_of_nodes()}, Edges: {s_net.number_of_edges()}")
    try:
        history = pd.read_csv("simulation_history.csv")
        print("Loaded existing simulation history.")
    except FileNotFoundError:
        history, h_post_exp, s_post_exp = run_experiment(h_net, s_net, steps=2000, alpha=0.3)
        history.to_csv("simulation_history.csv", index=False)
    print(history.tail())
    # plot_wealth_gap(history)
    plot_results(history)
    plot_lorenz_curve(h_post_exp, s_post_exp)

if __name__ == "__main__":
    main()