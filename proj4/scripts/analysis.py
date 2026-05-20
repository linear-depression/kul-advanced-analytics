"""
Epstein Aviation Subgraph Analysis
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
ACTIONS = BASE / "outputs" / "actions.csv"
METRICS = BASE / "outputs" / "graph metrics.csv"

actions = pd.read_csv(ACTIONS)
metrics = pd.read_csv(METRICS)

SEP = "-" * 60


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# 1. Subgraph size
section("1. Subgraph Size")
print(f"Unique Edges : {len(actions)}")
print(f"Nodes : {len(metrics)}")

# 2. Most frequent actions
section("2. Top 10 Most Frequent Actions")
top_actions = (
    actions["action"]
    .value_counts()
    .head(10)
    .reset_index()
    .rename(columns={"index": "Action", "action": "Count", "count": "Count"})
)
top_actions.columns = ["Action", "Count"]
print(top_actions.to_string(index=False))

# 3. Heaviest individual edges (by Weight column)
section("3. Top 10 Heaviest Edges (by Weight)")
top_weight = (
    actions[["Source", "Target", "Weight"]]
    .sort_values("Weight", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
print(top_weight.to_string(index=False))

# 4. Community sizes 
section("4. Community Sizes (top 10 by Modularity Class)")
community_sizes = (
    metrics["modularity_class"]
    .value_counts()
    .head(10)
    .reset_index()
)
community_sizes.columns = ["Modularity Class", "Node Count"]
print(community_sizes.to_string(index=False))

# 5. Top nodes by Degree 
section("5. Top 10 Nodes by Degree")
top_degree = (
    metrics[["Label", "Degree", "indegree", "outdegree", "betweenesscentrality", "modularity_class"]]
    .sort_values("Degree", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
print(top_degree.to_string(index=False))

# 6. Combined centrality metrics (top 20 by Degree) 
section("6. Combined Centrality Metrics (Top 20 by Degree)")
top_combined = (
    metrics[["Label", "Degree", "betweenesscentrality", "closnesscentrality", "eigencentrality", "Eccentricity", "modularity_class"]]
    .sort_values("Degree", ascending=False)
    .head(20)
    .reset_index(drop=True)
)
top_combined.columns = ["Node", "Degree", "Betweenness", "Closeness", "Eigenvector", "Eccentricity", "Community"]
print(top_combined.to_string(index=False))
