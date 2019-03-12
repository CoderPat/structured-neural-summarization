#!/usr/bin/env python
"""
Usage:
    convert2graph.py [options] INPUT_XML_ARTICLES_PATTERN SUMMARIES_FOLDER OUT_FILEPATH_PREFIX

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import os
import glob
import sys
import json
import gzip
import pdb
import codecs
import traceback
from typing import Callable, List

from docopt import docopt

from data.utils import load_xml
from parsers.naturallanguage.gigaword.loadgigacorpus import parse_sample


def parse_cnndm_file(filename: str, write_sample_callback: Callable, summaries_folder: str) -> None:
    def process_sample(location, sample):
        assert location[0][0] == 'root'
        assert location[1][0] == 'document'
        provenance = sample['docId']
        def get_summary_tokens(sample_data)-> List[str]:
            assert provenance.endswith('.article')
            with open(os.path.join(summaries_folder, provenance[:-len('.article')] + '.abstr')) as f:
                summary = f.read()
                return [t.strip() for t in summary.split() if len(t.strip()) > 0]

        parsed = parse_sample(sample, provenance, headline_getter=get_summary_tokens)
        if parsed is not None:
            write_sample_callback(parsed)
        return True
    load_xml(filename, depth=2, func=process_sample)

def run(args):
    with gzip.GzipFile(args['OUT_FILEPATH_PREFIX'] + '-inputs.jsonl.gz', 'wb') as text_input_file:
        with  gzip.GzipFile(args['OUT_FILEPATH_PREFIX'] + '-summaries.jsonl.gz', 'wb') as summaries_file:
            num_points = 0
            writer = codecs.getwriter('utf-8')
            for file_idx, file in enumerate(glob.glob(args['INPUT_XML_ARTICLES_PATTERN'])):
                if not file.endswith('.article.xml'): continue
                print('Loading %s...' % file)

                def write_sample(textsum):
                    if len(textsum.summary_sentence) <= 1:
                        print('Single word headline: %s. Ignoring...' % textsum.summary_sentence)
                        return
                    json.dump(textsum.main_text.to_graph_object(), writer(text_input_file))
                    writer(text_input_file).write('\n')

                    json.dump(textsum.summary_sentence, writer(summaries_file))
                    writer(summaries_file).write('\n')
                    nonlocal num_points
                    num_points += 1
                    if num_points % 1 == 0:
                        sys.stdout.write('\r\x1b[KLoaded %i files so far and %i samples.' % (file_idx + 1, num_points))
                        sys.stdout.flush()

                parse_cnndm_file(file, write_sample, args['SUMMARIES_FOLDER'])

    print('\n Finished loading %s datapoints.' % num_points)

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