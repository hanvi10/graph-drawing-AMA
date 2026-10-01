"""
scripts/download_dataset.py
===========================
Downloads the same graph dataset used
"Graph Drawing using Edge Relaxation".

HOW TO RUN:
    python3 scripts/download_dataset.py      (from the project root)

WHAT IT DOES:
    1. Asks the Netzschleuder website for a list of all available graphs.
    2. Keeps only the graphs that match the criteria.
    3. Downloads each graph, cleans it up, and saves it as a .graphml file.
    4. Saves a metadata CSV with one row per accepted graph.

OUTPUT FILES:
    data/graphs/<name>.graphml   -- one file per graph
    data/metadata.csv            -- table with name, tags, nodes, edges, initial_crossings
                                    (crossings in the Baseline drawing)
"""

import json      
import os        
import sys
import time      

import networkx as nx   
import pandas as pd     
import requests         
import zstandard as zstd  

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root (this file is in scripts/)
sys.path.insert(0, ROOT)   # so algorithms/ and metrics import from here

from algorithms.baseline import Baseline
from metrics import count_crossings


# ============================================================================
# SETTINGS
# ============================================================================

MIN_EDGES = 50     # discard graphs with fewer than this many edges
MAX_EDGES = 1000   # discard graphs with more than this many edges

# Graphs with any of these tags are excluded 
EXCLUDED_TAGS = {"Temporal", "Timestamps", "Multigraph"}

# URL of the Netzschleuder catalogue API (returns JSON with all graph info)
API_URL = "https://networks.skewed.de/api/nets?full=True"

# URL template to download a specific graph file
# {dataset} = top-level collection name, {net} = individual graph name
FILE_URL = "https://networks.skewed.de/net/{dataset}/files/{net}.gt.zst"

# Where to save the downloaded graphs
OUTPUT_DIR = os.path.join(ROOT, "data", "graphs")
METADATA_PATH = os.path.join(ROOT, "data", "metadata.csv")


# ============================================================================
# STEP 1 — PARSE THE GRAPH-TOOL BINARY FORMAT
# ============================================================================
# Netzschleuder stores graphs in graph-tool's custom binary format (.gt).
# Files are also compressed with zstandard (.zst).
# This function reads the raw bytes and builds a NetworkX graph.

def parse_gt_bytes(raw: bytes) -> nx.Graph:
    """
    Parse a graph-tool binary blob and return a NetworkX Graph.

    The .gt format stores:
      - A 6-byte magic header
      - Version number (1 byte)
      - Endianness flag (1 byte): whether numbers are big-endian or little-endian
      - A comment string
      - Whether the graph is directed (1 byte)
      - Number of nodes N (8 bytes)
      - For each node: how many neighbours it has, then their IDs
    """
    i = 0  # current reading position inside the byte array

    def read(nbytes, as_type=None, endian=None):
        """Read `nbytes` from `raw` starting at position `i`, then advance i."""
        nonlocal i
        chunk = raw[i: i + nbytes]
        i += nbytes
        if as_type == "str":   return chunk.decode()
        if as_type == "bool":  return chunk[0] == 1
        if as_type == "int":   return int.from_bytes(chunk, endian)
        return chunk  # return raw bytes if no conversion requested

    read(6, "str")                          # skip magic header "gt\x00..."
    read(1, "int", "little")               # skip version number
    endian = "big" if read(1, "bool") else "little"   # read endianness

    comment_length = read(8, "int", endian)
    read(comment_length, "str")             # skip the comment string

    directed = read(1, "bool")             # is this a directed graph?
    N = read(8, "int", endian)             # number of nodes

    # Create the appropriate graph type
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(N))

    # The number of bytes used per node ID depends on how many nodes there are
    # (smaller graphs can use fewer bytes to save space)
    if   N <= 2**8:  id_bytes = 1
    elif N <= 2**16: id_bytes = 2
    elif N <= 2**32: id_bytes = 4
    else:            id_bytes = 8

    # Read each node's neighbour list and add edges
    for src_node in range(N):
        num_neighbours = read(8, "int", endian)
        for _ in range(num_neighbours):
            dst_node = read(id_bytes, "int", endian)
            G.add_edge(src_node, dst_node)

    return G


def download_and_parse(dataset: str, net: str) -> nx.Graph:
    """Download a .gt.zst file from Netzschleuder and parse it."""
    url = FILE_URL.format(dataset=dataset, net=net)
    response = requests.get(url, timeout=60)
    response.raise_for_status()  # crash loudly if download failed

    raw = response.content

    # Decompress the .zst file into raw bytes
    with zstd.ZstdDecompressor().stream_reader(raw) as reader:
        raw = reader.read()

    return parse_gt_bytes(raw)


# ============================================================================
# STEP 2 — PREPROCESS THE GRAPH
# ============================================================================
# Three cleaning steps before using a graph:
#   a) Remove self-loops (an edge from a node to itself — can't be drawn)
#   b) If the graph is disconnected, keep only the largest piece
#   c) Re-check the edge count after cleaning

def preprocess(G: nx.Graph) -> nx.Graph | None:
    """
    Clean up a graph. Returns the cleaned graph, or None if it no longer
    meets the edge count requirement after cleaning.
    """
    # Make sure we're working with an undirected graph
    G = G.to_undirected()

    # (a) Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # (b) Keep only the largest connected component
    if not nx.is_connected(G):
        largest_component = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_component).copy()
        # Rename nodes to 0, 1, 2, ... so there are no gaps
        G = nx.convert_node_labels_to_integers(G)

    # (c) Check edge count again — the graph may have shrunk after cleaning
    if not (MIN_EDGES <= G.number_of_edges() <= MAX_EDGES):
        return None  # discard this graph

    return G


# ============================================================================
# STEP 3 — COUNT EDGE CROSSINGS IN THE BASELINE DRAWING
# ============================================================================
# Discards graphs whose initial drawing has 0 crossings: they are already
# drawn crossing-free, so edge relaxation can't improve them (the paper's
# rule). The initial drawing is our Baseline (spectral + ForceAtlas2 + min
# separation, algorithms/baseline.py), and crossings are counted with the
# same metrics.count_crossings the experiments use. The graph is relabelled
# 0..n-1 exactly as the experiment runners load it, so `initial_crossings`
# in the metadata equals the baseline's crossings in the results.

def baseline_crossings(G: nx.Graph) -> int:
    """Crossings in the Baseline drawing of G, as the experiments see it."""
    G = nx.convert_node_labels_to_integers(G)
    return count_crossings(G, Baseline().layout(G))


# ============================================================================
# STEP 4 — FETCH CANDIDATE LIST FROM THE API
# ============================================================================

def fetch_candidates() -> list[dict]:
    """
    Ask the Netzschleuder API for all available graphs and return those
    that pass first-level filters (edge count, directed, tags).
    The zero-crossings filter is applied later (after downloading).
    """
    print("Fetching graph catalogue from networks.skewed.de ...")
    response = requests.get(API_URL, timeout=120)
    response.raise_for_status()
    catalogue = json.loads(response.text)

    candidates = []

    for dataset_name, info in catalogue.items():
        # Skip huge collections (>100 sub-graphs) to keep things manageable
        if len(info["nets"]) > 100:
            continue

        tags = set(info.get("tags", []))

        # Skip if ANY excluded tag is present
        if tags & EXCLUDED_TAGS:
            continue

        nets = info["nets"]
        analyses = info["analyses"]

        # A dataset can contain one or more individual graphs ("nets")
        for net in nets:
            # Get the analysis stats for this specific graph
            if isinstance(analyses, dict) and "num_edges" in analyses:
                # Single-net dataset: analyses IS the stats dict
                a = analyses
            else:
                # Multi-net dataset: analyses is a dict keyed by net name
                a = analyses.get(net, {})

            if not a:
                continue

            # Undirected graphs only
            if a.get("is_directed", True):
                continue

            num_edges = a.get("num_edges", 0)
            if not (MIN_EDGES <= num_edges <= MAX_EDGES):
                continue

            # Build a human-readable name: "dataset" or "dataset/net"
            if len(nets) == 1:
                name = dataset_name
            else:
                name = f"{dataset_name}/{net}"

            candidates.append({
                "dataset": dataset_name,
                "net": net,
                "name": name,
                "tags": sorted(tags),
                "V": a.get("num_vertices", 0),
                "E": num_edges,
            })

    print(f"  → {len(candidates)} candidates after API-level filtering.")
    return candidates


# ============================================================================
# MAIN — TIE EVERYTHING TOGETHER
# ============================================================================

def main():
    # Create the output folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # If we already downloaded some graphs, load them so we can resume
    if os.path.exists(METADATA_PATH):
        existing = pd.read_csv(METADATA_PATH)
        done = set(existing["name"].tolist())
        rows = existing.to_dict("records")
        print(f"Resuming: {len(done)} graphs already downloaded.")
    else:
        done = set()
        rows = []

    # Get the list of graphs to download
    candidates = fetch_candidates()

    # Counters for the final summary
    n_accepted = 0
    n_skip_edgecount = 0
    n_skip_planar = 0
    n_skip_error = 0

    for idx, cand in enumerate(candidates):
        name = cand["name"]

        # Skip graphs we already have
        if name in done:
            continue

        print(f"[{idx+1}/{len(candidates)}] {name} ...", end=" ", flush=True)

        # --- Download ---
        try:
            G_raw = download_and_parse(cand["dataset"], cand["net"])
        except Exception as exc:
            print(f"ERROR: {exc}")
            n_skip_error += 1
            time.sleep(2)  # wait a moment before the next request
            continue

        # --- Preprocess ---
        G = preprocess(G_raw)
        if G is None:
            print("SKIPPED — edge count out of range after cleaning")
            n_skip_edgecount += 1
            continue

        # --- Check for zero crossings in the baseline drawing ---
        crossings = baseline_crossings(G)
        if crossings == 0:
            print("SKIPPED — 0 crossings in the baseline drawing")
            n_skip_planar += 1
            continue

        # --- Save the graph ---
        # Replace "/" in names with "__" so it works as a filename
        safe_name = name.replace("/", "__")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.graphml")
        nx.write_graphml(G, out_path)

        # Record metadata
        rows.append({
            "name": name,
            "safe_name": safe_name,
            "tags": json.dumps(cand["tags"]),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "initial_crossings": crossings,
        })
        done.add(name)
        n_accepted += 1
        print(f"OK  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{crossings} crossings)")

        # Save metadata after each graph so we can resume if interrupted
        pd.DataFrame(rows).to_csv(METADATA_PATH, index=False)

    # Final summary
    print()
    print("=" * 50)
    print(f"  Accepted:                  {n_accepted}")
    print(f"  Skipped (edge count):      {n_skip_edgecount}")
    print(f"  Skipped (0 crossings):     {n_skip_planar}")
    print(f"  Skipped (download error):  {n_skip_error}")
    print(f"  Total in dataset:          {len(rows)}")
    print(f"  Graphs saved to:           {OUTPUT_DIR}")
    print(f"  Metadata saved to:         {METADATA_PATH}")


if __name__ == "__main__":
    main()
