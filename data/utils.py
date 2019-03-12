import gzip
import codecs
import json
import pickle
from typing import Any, Iterator, Callable, Iterable

import xmltodict


def load_xml_gz(filename: str, func: Callable, depth: int) -> Any:
    with gzip.open(filename) as f:
        return xmltodict.parse(f, item_depth=depth, item_callback=func)

def load_xml(filename: str, func: Callable, depth: int) -> Any:
    with open(filename, 'rb') as f:
        return xmltodict.parse(f, item_depth=depth, item_callback=func)

def save_pickle_gz(data: Any, filename: str) -> None:
    with gzip.GzipFile(filename, 'wb') as outfile:
        pickle.dump(data, outfile)

def iteratate_jsonl_gz(filename: str) -> Iterator[Any]:
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        for line in reader(f):
            yield json.loads(line)

def save_jsonl_gz(filename:str, data: Iterable[Any])-> None:
    with gzip.GzipFile(filename, 'wb') as out_file:
        writer = codecs.getwriter('utf-8')
        for element in data:
            writer(out_file).write(json.dumps(element))
            writer(out_file).write('\n')

def load_gz_per_line(filename:str)-> Iterator[str]:
    reader = codecs.getreader('utf-8')
    with gzip.open(filename) as f:
        yield from reader(f)