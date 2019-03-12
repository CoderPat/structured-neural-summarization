#!/usr/bin/env python
"""
Usage:
    copystats.py [options] INPUTS_JSONL SUMMARIES_JSONL

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
from docopt import docopt
import pdb
import traceback
import sys

import numpy as np
from dpu_utils.utils import load_jsonl_gz


def run(args):
    num_toks_can_be_copied_to_summary = []
    num_toks_in_summary = []

    for graph, summary in zip(load_jsonl_gz(args['INPUTS_JSONL']), load_jsonl_gz(args['SUMMARIES_JSONL'])):
        summary_tokens = set(summary)
        graph_tokens = set(graph['node_labels'])
        num_toks_in_summary.append(len(summary_tokens))
        num_toks_can_be_copied_to_summary.append(len(graph_tokens & summary_tokens))

    num_toks_in_summary = np.array(num_toks_in_summary, dtype=np.float)
    num_toks_can_be_copied_to_summary = np.array(num_toks_can_be_copied_to_summary, dtype=np.float)

    print('===== Statistics ==== ')
    print('Avg %% of tokens that can be copied: %s%%' % float(100 * np.mean(num_toks_can_be_copied_to_summary / num_toks_in_summary)))

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
