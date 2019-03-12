"""
Usage:
    graph2otherformat.py [options] INPUTS_FILE REWRITTEN_OUTPUTS_FILE

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""

from docopt import docopt
import pdb
import sys
import traceback
from typing import Iterable, List, Dict
import re
from collections import defaultdict

from data.utils import iteratate_jsonl_gz, save_jsonl_gz
from parsers.sourcecode.utils import subtokenizer


IDENTIFIER_REGEX = re.compile(r'[a-zA-Z_][a-zA-Z_0-9]*')

def graph_transformer(initial_graph: Dict)-> Dict:
    all_edges = defaultdict(lambda : defaultdict(set))
    for edge_type, from_idx, to_idx in initial_graph['edges']:
        all_edges[edge_type][from_idx].add(to_idx)

    backbone_seq = initial_graph['backbone_sequence']  # type: List[int]
    nodes = initial_graph['node_labels']  # type: List[str]

    nodes_to_subtokenize = {}  # type: Dict[int, List[str]]
    for i, node_idx in enumerate(backbone_seq):
        if not IDENTIFIER_REGEX.match(nodes[node_idx]):
            continue
        subtokens = subtokenizer(nodes[node_idx])
        if len(subtokens) > 1:
            nodes_to_subtokenize[node_idx] = subtokens

    # Add subtoken nodes and related edges
    token_node_ids_to_subtoken_ids = defaultdict(list)  # type: Dict[int, List[int]]
    def add_node(node_name: str)-> int:
        idx = len(nodes)
        nodes.append(node_name)
        return idx

    for token_node_idx, subtokens in nodes_to_subtokenize.items():
        for subtoken in subtokens:
            subtoken_node_idx = add_node(subtoken)
            token_node_ids_to_subtoken_ids[token_node_idx].append(subtoken_node_idx)
            all_edges['Subtoken'][token_node_idx].add(subtoken_node_idx)

    # Now fix the backbone_sequence
    update_backbone_seq = []  # type: List[int]
    for node_idx in backbone_seq:
        if node_idx in token_node_ids_to_subtoken_ids:
            update_backbone_seq.extend(token_node_ids_to_subtoken_ids[node_idx])
        else:
            update_backbone_seq.append(node_idx)

    # Now make sure that there are NextToken edges
    for i in range(1, len(update_backbone_seq)):
        all_edges['NextToken'][update_backbone_seq[i-1]].add(update_backbone_seq[i])

    # Finally, output the defined data structure
    flattened_edges = []
    for edge_type, edges_of_type in all_edges.items():
        for from_idx, to_idxes in edges_of_type.items():
            for to_idx in to_idxes:
                flattened_edges.append((edge_type, from_idx, to_idx))

    return dict(edges=flattened_edges, node_labels=nodes, backbone_sequence=backbone_seq)


def transform_graphs(input_file: str)-> Iterable[Dict]:
    for graph in iteratate_jsonl_gz(input_file):
        yield graph_transformer(graph)

def run(args):
    save_jsonl_gz(args['REWRITTEN_OUTPUTS_FILE'], transform_graphs(args['INPUTS_FILE']))

if __name__ == '__main__':
    args = docopt(__doc__)
    try:
        run(args)
    except:
        if args.get('--debug', False):
            _, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise