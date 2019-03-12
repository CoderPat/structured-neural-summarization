"""
Usage:
    loadbarone.py [options] INPUT_CODE_DATA INPUT_DOCS_DATA OUT_FILE_PREFIX

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
    --task-type                The type of task to extract this. Either "func-doc" or "func-name". Defaults to "func-doc".
"""
from ast import parse
from collections import defaultdict
from docopt import docopt
import re
import json

from data.utils import save_jsonl_gz
from parsers.sourcecode.barone.ast_graph_generator import AstGraphGenerator


# Extracts function names
def decl_tokenizer(decl):
    function_name = re.search('(?<=def )[\w_-]+(?=\(.*\):)', decl).group(0)
    return splitter(function_name)


# Tokenize on spaces and around delimiters.
# Keep delimiters together only if it makes sense (e.g. parentheses, dots)
docstring_regex_tokenizer = re.compile(
    r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")


def docstring_tokenize(docstr: str):
    return [t for t in docstring_regex_tokenizer.findall(docstr) if t is not None and len(t) > 0]


def process_data(inputs, outputs, task_type):
    data, graph_node_labels, docs_words = [], [], []
    num_inits, errors = 0, 0
    doc_tokenizer = docstring_tokenize

    for idx, (inp, output) in enumerate(zip(inputs, outputs)):
        try:
            if idx % 100 == 0:
                print('%.1f %%    \r' % (idx / float(len(inputs)) * 100), end="")

            visitor = AstGraphGenerator()
            visitor.visit(parse(inp))

            edge_list = [(t, origin, destination)
                         for (origin, destination), edges
                         in visitor.graph.items() for t in edges]

            if task_type == "func-doc":
                docs_words.append(doc_tokenizer(output))
            if task_type == "body-name":
                docs_words.append(decl_tokenizer(output))

            graph_node_labels = [label.strip() for (_, label) in sorted(visitor.node_label.items())]

            data.append({"edges": edge_list,
                         "backbone_sequence": visitor.terminal_path,
                         "node_labels": graph_node_labels})

        except Exception as e:
            errors += 1

    print("Generated %d graphs out of %d snippets" %
          (len(inputs) - errors, len(inputs)))

    return data, docs_words


def main():
    args = docopt(__doc__)
    code_data = args['INPUT_CODE_DATA']
    docs_data = args['INPUT_DOCS_DATA']

    task_type = args.get('--task_type', "func-doc")

    with open(code_data) as f:
        inputs = f.readlines()
    with open(docs_data, 'rb') as f:
        labels = [line.decode(encoding='utf-8', errors='ignore') for line in f]

    inputs = [inp.replace("DCNL ", "\n").replace(
        " DCSP ", "\t").replace("DCSP ", "\t") for inp in inputs]

    # unident body so it can be parsed
    if task_type == 'func-name':
        inputs = ["\n".join([line[2 if not idx and line[1] == "\t" else 1:]
                             for idx, line in enumerate(inp.split("\n"))]) for inp in inputs]

    labels = [label.replace("DCNL ", "\n").replace("DCSP ", "\t") for label in labels]

    assert len(labels) == len(inputs)

    graphs, docs = process_data(inputs, labels, task_type)

    save_jsonl_gz(args['OUT_FILE_PREFIX'] + "_graphs.jsonl.gz", graphs)
    save_jsonl_gz(args['OUT_FILE_PREFIX'] + "_summary.jsonl.gz", docs)


if __name__ == "__main__":
    main()
