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
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Callable

from docopt import docopt

from data.utils import load_xml

from nltk.tree import Tree

from parsers.naturallanguage.graphtextrepr import (DependencyEdge,
                                                          GraphTextRepresentation,
                                                          Token)
from parsers.naturallanguage.textsummary import TextSummary
from data.utils import load_xml_gz


def parse_tree_to_sentence(parse_tree:str)-> List[str]:
    return Tree.fromstring(parse_tree).leaves()

def try_find_RB_span(tokens: List[str]) -> Optional[Tuple[int, int]]:
    try:
        lrb_idx = tokens.index('-LRB-')
        rrb_idx = tokens.index('-RRB-')
        if lrb_idx > rrb_idx:
            return None  # Malformed title, parentheses misplaced
        return (lrb_idx, rrb_idx+1)
    except ValueError:
        return None


def parse_sample(datapoint, provenance: str, headline_getter: Optional[Callable[[Any], List[str]]]=None)-> Optional[TextSummary]:
    if headline_getter is None and (datapoint.get('HEADLINE') is None or len(datapoint['HEADLINE']) == 0):
        return None
    try:
        if headline_getter is None:
            headline_tokens = parse_tree_to_sentence(datapoint['HEADLINE'])
        else:
            headline_tokens = headline_getter(datapoint)
        # Remove LRB-RRB chunks
        rb_span = try_find_RB_span(headline_tokens)
        while rb_span is not None:
            headline_tokens = headline_tokens[:rb_span[0]] + headline_tokens[rb_span[1]:]
            rb_span = try_find_RB_span(headline_tokens)
        if len(headline_tokens) <= 1:
            return None

    except Exception as e:
        print('Could not parse %s. Ignoring sample.' % datapoint.get('HEADLINE'))
        print(e)
        return None

    if 'sentences' not in datapoint or datapoint['sentences'] is None:
        return None

    all_sentences = datapoint['sentences']['sentence']
    if type(all_sentences) is not list:
        all_sentences = [all_sentences]

    tokenized_sentences = []  # type: List[List[Token]]
    for sentence in all_sentences:
        sentence_tokens = []
        if type(sentence['tokens']['token']) is not list:
            # May happen in single-word sentences
            sentence['tokens']['token'] = [sentence['tokens']['token']]
        for i, token in enumerate(sentence['tokens']['token']):
            assert int(token['@id']) == i + 1
            sentence_tokens.append(Token(word=token['word'], lemma=token['lemma'], pos_tag=token['POS']))
        tokenized_sentences.append(sentence_tokens)

    graph_text_representation = GraphTextRepresentation(tokenized_sentences, provenance=provenance)

    # Add named entities, by finding consecutive annotations
    for sentence_idx, sentence in enumerate(all_sentences):
        sentence_tokens = sentence['tokens']['token']
        for token_idx, token in enumerate(sentence_tokens):
            if 'NER' not in token:
                return None  # Ignore samples that don't have NER output.
            if token['NER'] == 'O':
                continue
            if token_idx + 1 < len(sentence_tokens) - 1 and sentence_tokens[token_idx + 1]['NER'] != token['NER']:
                # Create an entity that includes this token as the last one
                before_start_token_idx = token_idx - 1
                while before_start_token_idx > 0 and sentence_tokens[before_start_token_idx]['NER'] == token['NER']:
                    before_start_token_idx -= 1
                graph_text_representation.add_entity(token['NER'], sentence_idx, before_start_token_idx + 1, token_idx + 1)

    def get_collapsed_dependencies(sentence):
        if 'dependencies' not in sentence or sentence['dependencies'] is None:
            return None
        for dependencies in sentence['dependencies']:
            if dependencies['@type'] == 'collapsed-dependencies':
                return dependencies
        return None

    # Add dependencies
    for sentence_idx, sentence in enumerate(all_sentences):
        if ('collapsed-dependencies' not in sentence or sentence['collapsed-dependencies'] is None) and get_collapsed_dependencies(sentence) is None:
            continue
        if 'collapsed-dependencies' in sentence:
            collapsed_deps = sentence['collapsed-dependencies']
        else:
            collapsed_deps = get_collapsed_dependencies(sentence)

        if type(collapsed_deps['dep']) is not list:
            collapsed_deps['dep'] = [collapsed_deps['dep']]
        for dependency in collapsed_deps['dep']:
            if dependency['@type'] == 'root':
                continue  # Root is not useful for us
            dependency_type = dependency['@type']
            underscore_location = dependency_type.find('_')
            if underscore_location != -1:
                dependency_type = dependency_type[:underscore_location]
            if isinstance(dependency['dependent'], OrderedDict):
                dependency['dependent'] = dependency['dependent']['@idx']
            if isinstance(dependency['governor'], OrderedDict):
                dependency['governor'] = dependency['governor']['@idx']

            graph_text_representation.add_dependency_edge(DependencyEdge(
                dependency_type=dependency_type,
                sentence_idx=sentence_idx,
                from_idx=int(dependency['dependent']) - 1,
                to_idx=int(dependency['governor']) - 1
            ))

    # Add co-references
    coreferences = None
    if 'coreferences' in datapoint and datapoint['coreferences'] is not None:
        coreferences = datapoint['coreferences']
    elif 'coreference' in datapoint and datapoint['coreference'] is not None:
        coreferences = datapoint['coreference']

    if coreferences is not None:
        if type(coreferences['coreference']) is not list:
            coreferences['coreference'] = [coreferences['coreference']]
        for coreference in coreferences['coreference']:
            all_mentions = coreference['mention']
            representative = [m for m in all_mentions if '@representative' in m and m['@representative'] == 'true'][0]

            for mention in all_mentions:
                if mention.get('@representative') == 'true' or (mention['sentence'] == representative['sentence'] and mention['head'] == representative['head']):
                    continue
                graph_text_representation.add_coreference(int(mention['sentence']) - 1, int(mention['head']) - 1,
                                                          int(representative['sentence']) -1, int(representative['head'])-1)

    return TextSummary(
        summary_sentence=headline_tokens,
        main_text= graph_text_representation
    )

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