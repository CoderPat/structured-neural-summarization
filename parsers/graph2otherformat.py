#!/usr/bin/env python
"""
Usage:
    graph2otherformat.py [options] INPUTS_FILE SUMMARIES_FILE (opennmt|gettothepoint) OUTPUT_PREFIX

Options:
    --max-chunk-size=<size>    The max size of each chunk for getothepoint [default: 1000]
    --chain-subtokenization    If this is set, tokens are subtokenized and each subtoken is a token
    --vocab                    If this is set for computing a vocabulary for gettothepoint
    --max-vocab-size=<size>    The maximum size of the vocabulary for gettothepoint [default: 200000]
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""

from docopt import docopt
import pdb
import sys
import struct
import traceback
from typing import Iterable, Tuple, Callable, List, Dict, TypeVar
from collections import Counter
from itertools import chain

from data.utils import iteratate_jsonl_gz
from parsers.sourcecode.utils import subtokenizer as code_subtokenizer

from tensorflow.core.example import example_pb2

T = TypeVar('T')

def opennmt_converter(graph:Dict, summary: List[str], subtokenizer=None) -> Tuple[str, str]:
    if subtokenizer is None:
        def subtokenizer(token):
            return [token]
    
    input_sequence = chain.from_iterable(subtokenizer(graph['node_labels'][i]) for i in graph['backbone_sequence'])
    return ' '.join(w.lower() for w in input_sequence), ' '.join(w.lower() for w in summary)

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
GET_TO_THE_POINT_END_TOKENS = {'.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"} # acceptable ways to end a sentence


def gettothepoint_converter():
    vocab_counter = Counter()

    def converter(graph:Dict, summary: List[str])-> str:
        input_sequence = [graph['node_labels'][i].lower() for i in graph['backbone_sequence']]
        assert len(input_sequence) > 0
        assert len(summary) > 0

        summary = [w.lower() for w in summary]
        if summary[-1] not in GET_TO_THE_POINT_END_TOKENS:
            summary.append('.')

        vocab_counter.update(input_sequence)
        vocab_counter.update(summary)

        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([' '.join(input_sequence).encode()])
        tf_example.features.feature['abstract'].bytes_list.value.extend([('<s>'+ ' '.join(summary) + '</s>').encode()])
        return tf_example.SerializeToString()

    return vocab_counter, converter

def convert_dataset(inputs_filename: str, summaries_filename: str,
                    sample_converter: Callable[[Dict, List[str]], T],
                    subtokenizer)-> Iterable[T]:
    for input_graph, output_text in zip(iteratate_jsonl_gz(inputs_filename), iteratate_jsonl_gz(summaries_filename)):
        yield sample_converter(input_graph, output_text, subtokenizer)

def transform_to_opennmt(args):
    with open(args['OUTPUT_PREFIX'] + '-inputs.txt', 'w') as inputs:
        with open(args['OUTPUT_PREFIX'] + '-targets.txt', 'w') as targets:
            if args['--chain-subtokenization']:
                subtokenizer = code_subtokenizer
            else:
                subtokenizer = None

            for text, summary in convert_dataset(args['INPUTS_FILE'], args['SUMMARIES_FILE'], opennmt_converter, subtokenizer):
                inputs.write(text)
                inputs.write('\n')

                targets.write(summary)
                targets.write('\n')

def transform_to_gettothepoint(args):
    num_chunks = 0
    num_elements_in_chunk = 0
    max_chunk_size = int(args['--max-chunk-size'])

    out_chunk = open(args['OUTPUT_PREFIX'] + '_%03d.bin' % num_chunks, 'wb')

    vocabulary, datapoint_converter = gettothepoint_converter()

    for datapoint in convert_dataset(args['INPUTS_FILE'], args['SUMMARIES_FILE'], datapoint_converter):
        length = len(datapoint)
        out_chunk.write(struct.pack('q', length))
        out_chunk.write(struct.pack('%ds' % length, datapoint))
        num_elements_in_chunk += 1

        if num_elements_in_chunk >= max_chunk_size:
            out_chunk.close()
            num_chunks += 1
            num_elements_in_chunk = 0
            out_chunk = open(args['OUTPUT_PREFIX'] + '_%03d.bin' % num_chunks, 'wb')

    out_chunk.close()

    if args['--vocab']:
        with open(args['OUTPUT_PREFIX'] + '-vocab', 'w') as writer:
            for word, count in vocabulary.most_common(int(args['--max-vocab-size'])):
                if ' ' in word or '\\/' in word: continue
                writer.write(word + ' ' + str(count) + '\n')


def run(args):
    if args['opennmt']:
        transform_to_opennmt(args)
    elif args['gettothepoint']:
        transform_to_gettothepoint(args)
    else:
        raise Exception('No recognized target format.')





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
