#!/usr/bin/env python
"""
Usage:
    visualizegraphs.py [options] GRAPH_FILE TARGET_FILE

Options:
    --edges-to-exclude=<val>   Comma-separated list of edge types to exclude
    --edges-to-include=<val>   Comma-separated list of edge types to exclude
    --outfile OUTFILE          The file to output the graph [default: graph.dot]
    --up-to-nodeid=<val>         Use all nodes in the backbone sequence up to the given value
    -h --help                  Show this screen.

"""
import json
from docopt import docopt


if __name__ == '__main__':
    arguments = docopt(__doc__)


    # Search for target
    with open(arguments['GRAPH_FILE']) as f:
        target_graph = json.load(f)

    def escape_dot(label: str) ->str:
        return label.replace('"','\\"')

    # Precompute info
    with open(arguments.get('TARGET_FILE'), 'w') as f:
        f.write('## Graph for %s:%s\n' % (arguments['GRAPH_FILE'], target_graph['provenance']))
        f.write('digraph ProgramGraph {\n')

        f.write('\tgraph[ fontname = "Consolas"];\n')
        f.write('\tnode [shape = box, style="filled,solid", fontname = "Consolas"];\n')
        f.write('\n')

        if arguments.get('--up-to-nodeid') is None:
            excluded_node_ids = set()
        else:
            excluded_node_ids = set(target_graph['backbone_sequence'])
            for i in range(int(arguments.get('--up-to-nodeid'))):
                excluded_node_ids.remove(i)

        for node_id, label in enumerate(target_graph['node_labels']):
            if node_id in excluded_node_ids:
                continue
            if label == "Sentence":
                fillcolor = 'red'
            elif label.startswith("ENT:"):
                fillcolor = 'yellow'
            else:
                fillcolor = None
            if fillcolor is None:
                f.write('\tn%s[label="%s"];\n' % (node_id, escape_dot(label)))
            else:
                f.write('\tn%s[label="%s", color="%s"];\n' % (node_id, escape_dot(label), fillcolor))

        excludedEdgeTypes = None
        includedEdgeTypes = None
        if arguments.get('--edges-to-exclude') is not None:
            assert arguments.get('--edges-to-include') is None, "--edges-to-exclude and --edges-to-include are mutually exclusive."
            excludedEdgeTypes = set(edge_type.strip() for edge_type in arguments['--edges-to-exclude'].split(','))
            print('Excluding edge types: %s' % excludedEdgeTypes)
        elif arguments.get('--edges-to-include') is not None:
            includedEdgeTypes = set(edge_type.strip() for edge_type in arguments['--edges-to-include'].split(','))
            print('Included edge types: %s' % includedEdgeTypes)



        for edge_type, from_node_id, to_node_id in target_graph['edges']:
            if from_node_id in excluded_node_ids or to_node_id in excluded_node_ids:
                continue
            if excludedEdgeTypes is not None and edge_type in excludedEdgeTypes:
                continue
            if includedEdgeTypes is not None and edge_type not in includedEdgeTypes:
                continue
            f.write('\tn%s -> n%s [label="%s"];\n' % (from_node_id, to_node_id, escape_dot(edge_type)) )
        f.write('}\n')
