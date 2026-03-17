"""Utility to construct a simple knowledge graph from the
`weighted_triplets.csv` produced by the pipeline.

This example uses networkx for graph construction and optionally
writes out a visualisation via pyvis (or other library).

Usage:
    python build_graph.py \
        --input /scratch/data/r24ab0001/kg_outputs/weighted_triplets.csv \
        --out-graph graph.gpickle --out-html graph.html

The resulting graph has nodes keyed by CUI, each node carrying the
canonical name as an attribute; edges are (subj, obj) annotated with
`predicate`, `assertion` and `weight`.
"""

import argparse
import pandas as pd
import networkx as nx


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Convert the DataFrame of weighted triplets into a directed multigraph.

    We use a MultiDiGraph so that multiple predicates between the same
    two CUIs can coexist (e.g. `location_of` and `associated_with`).
    """
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        subj = row['Subject_CUI']
        obj = row['Object_CUI']
        # add nodes with name attribute (will not overwrite once set)
        if subj not in G:
            G.add_node(subj, name=row['Subject_Name'])
        if obj not in G:
            G.add_node(obj, name=row['Object_Name'])
        # add edge
        G.add_edge(
            subj,
            obj,
            predicate=row['Predicate'],
            assertion=row['Assertion'],
            weight=row['Weight (Frequency)']
        )
    return G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='/scratch/data/r24ab0001/kg_outputs/weighted_triplets.csv',
                        help='CSV file with weighted triplets')
    parser.add_argument('--out-graph', '-g', default='/scratch/data/r24ab0001/kg_outputs/kg_graph.gpickle',
                        help='path to save networkx graph (pickle)')
    parser.add_argument('--out-gexf', '-x', default='/scratch/data/r24ab0001/kg_outputs/kg_graph.gexf',
                        help='optional path to save GEXF graph (Gephi)')
    parser.add_argument('--out-html', '-o', default='/scratch/data/r24ab0001/kg_outputs/kg_graph.html',
                        help='optional pyvis HTML visualisation file')
    parser.add_argument('--no-html', action='store_true',
                        help='do not attempt HTML visualization (skip pyvis)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} weighted triplets from {args.input}")

    graph = build_graph(df)
    print(f"Graph contains {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # networkx 3.x removed write_gpickle helper; use readwrite.gpickle instead
    try:
        nx.write_gpickle(graph, args.out_graph)
    except AttributeError:
        # networkx no longer provides write_gpickle; fall back to plain pickle
        import pickle
        with open(args.out_graph, 'wb') as f:
            pickle.dump(graph, f)
    print(f"Graph saved to {args.out_graph}")

    if args.out_gexf:
        try:
            nx.write_gexf(graph, args.out_gexf)
            print(f"GEXF graph saved to {args.out_gexf}")
        except Exception as e:
            print(f"Failed to write GEXF: {e}")

    if args.out_html and not args.no_html:
        try:
            from pyvis.network import Network
        except ImportError:
            print("pyvis not installed; skipping HTML output")
        else:
            net = Network(height="800px", width="100%", directed=True)
            # add nodes
            for n, data in graph.nodes(data=True):
                net.add_node(n, label=data.get('name', n))
            # add edges; convert multigraph to simple for visualization
            for u, v, attrs in graph.edges(data=True):
                label = attrs.get('predicate')
                if 'weight' in attrs:
                    label = f"{label} ({attrs['weight']})"
                net.add_edge(u, v, label=label)
            try:
                # pyvis.Network.show defaults to notebook=True, which leaves
                # `self.template` unset and raises an AttributeError when
                # generate_html tries to render it.  Use write_html directly
                # with notebook=False (static output) instead.
                net.write_html(args.out_html, notebook=False)
                print(f"HTML visualization saved to {args.out_html}")
            except Exception as e:
                print(f"Failed to generate HTML visualization: {e}")
    elif args.no_html:
        print("Skipping HTML visualization because --no-html was passed.")


if __name__ == '__main__':
    main()
