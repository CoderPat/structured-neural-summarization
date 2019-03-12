#!/usr/bin/env python
"""
Usage:
    graphstats.py [options] INPUTS_JSONL

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
from docopt import docopt
import pdb
import traceback
import sys

import numpy as np
from data.utils import iteratate_jsonl_gz


def run(args):
    num_samples = 0
    total_num_nodes = 0
    total_num_edges = 0
    sum_input_sequences_length = 0

    for sample in iteratate_jsonl_gz(args['INPUTS_JSONL']):
        num_samples += 1
        total_num_nodes += len(sample['node_labels'])
        total_num_edges += len(sample['edges'])
        sum_input_sequences_length += len(sample['backbone_sequence'])

    print('===== Statistics ==== ')
    print('Num samples: %i' % num_samples)
    print('Avg num nodes: %.2f' % (total_num_nodes / num_samples))
    print('Avg num edges: %.2f' % (total_num_edges / num_samples))
    print('Avg input seq length: %.2f' % (sum_input_sequences_length / num_samples))

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
