# Project 4 — Graph Analysis of Jeffrey Epstein's Aviation Criminal Network

A graph-theoretic study of the aviation subnetwork extracted from the Jeffrey Epstein legal document corpus. The project loads a large knowledge graph into Memgraph, isolates flight-related edges via a Cypher query, exports the subgraph to Gephi for community detection and layout, and performs quantitative metric analysis in Python.

---

## Repository Structure

```
proj4/
├── data/
│   ├── memgraph-nodes.csv     # Node data for bulk import into Memgraph
│   └── memgraph-edges.csv     # Edge data for bulk import into Memgraph
├── outputs/
│   ├── actions.csv            # Exported aviation subgraph edges 
│   ├── graph metrics.csv      # Per-node centrality metrics exported from Gephi
│   └── Gephi project 1.gephi  # Gephi project with ForceAtlas2 layout and Louvain communities
├── plots/
│   ├── community.png          # Full network visualization (all communities)
│   ├── blue.png               # Blue community — Core Epstein network
│   ├── green.png              # Green community — Legal cluster
│   ├── purple.png             # Purple community — Pilot operations
│   └── yellow.png             # Yellow community — VIP passengers
├── report/
│   └── graph_analysis_report.pdf  # Compiled PDF report
└── scripts/
    └── analysis.py            # Python script for quantitative metric analysis
```

