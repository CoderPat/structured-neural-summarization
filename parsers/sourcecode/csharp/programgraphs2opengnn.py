#!/usr/bin/env python
"""
Usage:
    programgraphs2opengnn.py [options] INPUT_FILES_PATTERN OUTPUT_PREFIX

Options:
    --names                    If enabled, the summaries will consist of the function names rather than actual summaries.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import os
import gzip
import pdb
import re
import json
import sys
import glob
import traceback
import codecs
import multiprocessing as mp
import tqdm

from docopt import docopt, DocoptExit
from collections import Counter, OrderedDict, Iterator

from parsers.sourcecode.utils import extract_path, subtokenizer

def load_json_gz(filename: str):
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        return json.load(reader(f), object_pairs_hook=OrderedDict)

# threshold to ignore methods that share same docstring among themselves
# avoids biasing the dataset with commonly overloaded methods (ie: ==)
MAX_DUPLICATE_DOCSTRINGS = 5


def docstring_tokenize(docstr: str):
    # Tokenize on spaces and around delimiters.
    # Keep delimiters together only if it makes sense (e.g. parentheses, dots)
    # TODO: Regex might need change since it was originially for python
    xml_parser = re.compile(r'<.*ref=["\']([^\'"]+)["\'] */>')
    docstring_regex_tokenizer = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")
    return [t for t in docstring_regex_tokenizer.findall(xml_parser.sub(r"\1", docstr)) if t is not None and len(t) > 0]


def process_sample(sample, names):
    graph = {'edges': []}
    raw_graph = sample['Graph']
    # transform from edge list for each edge_type to list of all edges
    for edge_type, edges in raw_graph['Edges'].items():
        for edge in edges:
            graph['edges'].append([edge_type] + edge)

    labels = raw_graph['NodeLabels'].values() if isinstance(raw_graph['NodeLabels'], dict) else raw_graph['NodeLabels']

    if all([token.isspace() for token in labels]):
        return None

    graph['node_labels'] = [token if token != "DECLARATION" or names else sample['Name'] for token in labels]
    graph['backbone_sequence'] = extract_path(graph['edges'], 'NextToken')

    if names:
        summary = subtokenizer(sample['Name'])
    else:
        # tokenize docstring
        if 'Summary' not in sample:
            return None

        summary = docstring_tokenize(sample['Summary'])

    if all([token.isspace() for token in summary]):
        return None

    return graph, summary

def process_file(filename):
    samples_in_file = load_json_gz(filename)
    results = []
    for sample in samples_in_file:
        results.append(process_sample(sample, args.get('--names', False)))
    return results

def process_all_samples(input_files_pattern: str)-> Iterator:
    all_files = glob.glob(input_files_pattern)
    with mp.Pool() as p:
        for processed_samples in p.imap_unordered(process_file, all_files, chunksize=1):
            yield from processed_samples

def run(args):
    input_pattern = args['INPUT_FILES_PATTERN']
    output_prefix = args['OUTPUT_PREFIX']

    summary_counter = Counter()
    def graph_and_summary_iter():
        for sample in process_all_samples(input_pattern):
            if sample is None:
                continue
            summary_counter[tuple(sample[1])] += 1
            yield sample[0], sample[1]

    # Write to OpenGNN format
    total_counter, valid_counter = 0, 0

    with gzip.open("%s_graphs.jsonl.gz" % output_prefix, 'wb') as graph_output_file:
        with gzip.open("%s_summaries.jsonl.gz" % output_prefix, 'wb') as summary_output_file:
            for graph, summary in tqdm.tqdm(graph_and_summary_iter()):
                total_counter += 1
                if summary_counter[tuple(summary)] > MAX_DUPLICATE_DOCSTRINGS:
                    continue
                graph_output_file.write(("%s\n" % json.dumps(graph)).encode('utf-8'))
                summary_output_file.write(("%s\n" % json.dumps(summary)).encode('utf-8'))
                valid_counter += 1

    print('Processed %s. Only %s were added.' % (total_counter, valid_counter))

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
