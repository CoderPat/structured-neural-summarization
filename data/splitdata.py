#!/usr/bin/env python
"""
Usage:
    splitdata.py [options] INPUTS_FILE SUMMARIES_FILE OUTPUT_PATH FOLDS_SPEC 

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
    --shard_files LIMIT        If enables divides samples of each split into multiple files, each with maximum LIMIT samples     
"""
import codecs
import gzip
import json
import os
import pdb
import random
import sys
import traceback

from collections.__init__ import OrderedDict
from typing import Dict, Optional, Iterator, Generator, Tuple

from docopt import docopt

from data.utils import load_gz_per_line

def get_prefix(path):
    for ext in ['.gz', '.tar.gz']:
        if path.endswith(ext):
            path = path[:-len(ext)]
    return os.path.splitext(path)[0]

class SplitFileWriter:
    def __init__(self,
                 output_filename_prefix: str,
                 output_filename_suffix: str = 'json.gz', 
                 max_elements_per_file: Optional[int] = None):
        self.__output_filename_prefix = output_filename_prefix
        self.__output_filename_suffix = output_filename_suffix
        self.__num_files_written = 0
        self.__max_elements_per_file = max_elements_per_file
        self.__current_file = None
        self.__curr_file_size = 0

    def generate_file_name(self):
        if self.__max_elements_per_file is not None:
            return ('%s.%s.%s' % (self.__output_filename_prefix,
                                  self.__num_files_written,
                                  self.__output_filename_suffix))
        else:
            return ('%s.%s' % (self.__output_filename_prefix,
                               self.__output_filename_suffix))

    def _create_new_file(self):
        filename = self.generate_file_name()
        self.__current_file = gzip.open(filename, 'w')
        self.__num_files_written += 1

    def _close_current_file(self):
        self.__current_file.close()
        self.__current_file = None
        self.__curr_file_size = 0

    def add(self, element):
        if self.__current_file is None:
            self._create_new_file()

        self.__current_file.write(bytes(element, 'utf-8'))
        self.__curr_file_size += 1

        if (self.__max_elements_per_file is not None and
                self.__curr_file_size > self.__max_elements_per_file):
            self._close_current_file()

    def close(self):
        if self.__current_file is not None:
            self._close_current_file()


def split_jsonl_gz(inputs_filepath: str, summaries_filepath: str,
                   output_path: str, fold_proportions: Dict[str, float], 
                   seed: Optional[int]=None, max_elements_per_file: Optional[int]=None)-> None:
    assert abs(sum(fold_proportions.values()) - 1) < 1e-5, 'Fold proportions must sum to 1.'
    assert len(fold_proportions) > 0

    thresholds = OrderedDict()  # type: OrderedDict[str, float]
    proportion_accumulator = 0.0
    for fold, proportion in fold_proportions.items():
        os.makedirs(os.path.join(output_path, fold), exist_ok=True)
        proportion_accumulator += proportion
        thresholds[fold] = proportion_accumulator

    def allocate_to_fold()-> str:
        rand_num = random.random()
        for fold, threshold in thresholds.items():
            if rand_num < threshold:
                return fold
        return fold  # This may happen if because of precision max(threshold.values()) < 1

    if seed is not None:
        random.seed(seed)

    out_files = {}  # type: Dict[str, Tuple[Generator[None, str, None], Generator[None, str, None]]]
    for fold in fold_proportions:
        fold_path = os.path.join(output_path, fold)

        inputs_fold_writer = SplitFileWriter(
            os.path.join(fold_path, get_prefix(os.path.basename(inputs_filepath))),
            max_elements_per_file=max_elements_per_file)

        outputs_fold_writer = SplitFileWriter(
            os.path.join(fold_path, get_prefix(os.path.basename(summaries_filepath))),
            max_elements_per_file=max_elements_per_file)

        out_files[fold] = inputs_fold_writer, outputs_fold_writer

    for input_line, summary_line in \
            zip(load_gz_per_line(inputs_filepath), load_gz_per_line(summaries_filepath)):
        fold = allocate_to_fold()
        inputs_writer, outputs_writer = out_files[fold]
        inputs_writer.add(input_line)
        outputs_writer.add(summary_line)

    # Close the files
    for inputs_fold_file, outputs_fold_file in out_files.values():
        inputs_fold_file.close()
        outputs_fold_file.close()


def run(args):
    shard_files = int(args.get('--shard_files')) if args.get('--shard_files') is not None else None
    split_jsonl_gz(args['INPUTS_FILE'], args['SUMMARIES_FILE'], args['OUTPUT_PATH'],
                   fold_proportions=json.loads(args['FOLDS_SPEC']),
                   max_elements_per_file=shard_files)


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
