#!/usr/bin/env python
"""
Usage:
    stripsemanticsfromgraphs.py [options] INPUT_GRAPH_JSONL_GZ OUTPUT_TARGET_JSONL_GZ

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
    --add-eq-edges             Add "eq" edges in the output graphs
"""
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from docopt import docopt
from dpu_utils.utils import run_and_debug

from data.utils import iteratate_jsonl_gz, save_jsonl_gz

ACCEPTED_EDGES = {"in", "next"}

EN_STOPWORDS = set(stopwords.words('english'))
STEMMER = SnowballStemmer('english')

def convert_graph(graph_data: Dict[str, Any], add_eq_edges: bool)-> Dict[str, Any]:
    """Convert graph."""
    new_graph = {}  # type: Dict[str, Any]
    old_backbone_node_ids = set(graph_data['backbone_sequence'])

    # Remap ids
    old_to_new_id = OrderedDict()  # type: OrderedDict[int, int]
    for i, node_label in enumerate(graph_data['node_labels']):
        if i in old_backbone_node_ids or node_label=='Sentence':
            old_to_new_id[i] = len(old_to_new_id)

    new_graph['node_labels'] = [graph_data['node_labels'][old_id] for old_id, new_id in old_to_new_id.items()]
    new_graph['backbone_sequence'] = [old_to_new_id[old_id] for old_id in graph_data['backbone_sequence']]

    new_graph['edges'] = [(e[0], old_to_new_id[e[1]], old_to_new_id[e[2]]) for e in graph_data['edges']
                          if e[0] in ACCEPTED_EDGES and e[1] in old_to_new_id and e[2] in old_to_new_id]

    new_graph['provenance'] = graph_data['provenance']

    if add_eq_edges:
        stemmed_node_ids = defaultdict(list)  # type: Dict[str, List[int]]
        for i, node_label in enumerate(new_graph['node_labels']):
            if node_label.lower() in EN_STOPWORDS or len(node_label) <= 1 :
                continue
            stemmed_node_ids[STEMMER.stem(node_label)].append(i)

        for eq_nodes in stemmed_node_ids.values():
            if len(eq_nodes) <= 1:
                continue
            new_graph['edges'].extend((('eq', eq_nodes[0], other_node_id) for other_node_id in eq_nodes[1:]))

    return new_graph

def run(arguments):
    save_jsonl_gz(arguments['OUTPUT_TARGET_JSONL_GZ'],
        data=(convert_graph(g, arguments['--add-eq-edges'])
              for g in iteratate_jsonl_gz(arguments['INPUT_GRAPH_JSONL_GZ'])))


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
