#!/usr/bin/env python
"""
Usage:
    splitfiles.py [options] FOLD_SPEC_PREFIX STORIES_DIR OUT_FOLDER

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import os
import nltk
from docopt import docopt
from typing import Iterable, List, Optional, Tuple, Set, Dict
import hashlib

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def load_split(prefix: str)-> Dict[str, Set[str]]:
    file_folds = {
        'train': set(),
        'val': set(),
        'test': set()
    }

    for fold in file_folds:
        with open(prefix + fold + '.txt') as f:
            for line in f:
                line = line.strip()
                if len(line)==0:
                    continue
                h = hashlib.sha1()
                h.update(line.encode())
                file_folds[fold].add(h.hexdigest())
    return file_folds

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence


def fix_missing_period(line: str)-> str:
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."


def read_text_file(text_file: str)-> List[str]:
    lines = []
    with open(text_file, 'r') as f:
        for line in f:
            if len(line.strip()) > 0:
                lines.append(line.strip())
    return lines

def get_art_abs(story_file)-> Tuple[str, str]:
    lines = read_text_file(story_file)

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == '':
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a single string, putting <s> and </s> tags around the sentences
    abstract = ' '.join([' '.join(nltk.word_tokenize(sent)) for sent in highlights])
    return article, abstract


if __name__ == '__main__':
    args = docopt(__doc__)

    split_spec = load_split(args['FOLD_SPEC_PREFIX'])
    for fold, story_ids in split_spec.items():
        fold_out_articles_path = os.path.join(args['OUT_FOLDER'], fold, 'articles')
        fold_out_summaries_path = os.path.join(args['OUT_FOLDER'], fold, 'summaries')
        os.makedirs(fold_out_articles_path, exist_ok=True)
        os.makedirs(fold_out_summaries_path, exist_ok=True)

        filelist_path = os.path.join(args['OUT_FOLDER'], fold+'_filelist.txt')
        story_files = []

        for story_id in story_ids:
            story_filepath = os.path.join(args['STORIES_DIR'], story_id + '.story')
            assert os.path.exists(story_filepath)
            article, abstract = get_art_abs(story_filepath)

            article_path = os.path.join(fold_out_articles_path, story_id + '.article')
            with open(article_path, 'w') as f:
                f.write(article+'\n')
            story_files.append(article_path)

            abstract_path = os.path.join(fold_out_summaries_path, story_id + '.abstr')
            with open(abstract_path, 'w') as f:
                f.write(abstract+'\n')
        with open(filelist_path, 'w') as f:
            f.write('\n'.join(story_files))
