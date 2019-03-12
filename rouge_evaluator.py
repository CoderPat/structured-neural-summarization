"""
Usage:
    rouge_evaluator [options] REFERENCES_FILE PREDICTIONS_FILE

Options:
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
    --format FORMAT            Format of the input data [jsonl, text, textfolder]. [default: jsonl]
    --case_sensitive           If enabled, considers casing of words. Defaults to false
    --use-rouge155             Use the standard ROUGE wrapper
"""
import logging
import os
from docopt import docopt
import json
import pdb
import traceback
import gzip
import tempfile

def open_file(filename):
    # TODO: better checks for filetype
    if filename.endswith('.gz'):
        return gzip.open(filename, mode='rt')
    else:
        return open(filename, mode='rt')


def extract_sentences(file, file_type, case_sensitive):
    sentences = []
    with open_file(file) as f:
        for sentence in f:
            if file_type == 'jsonl':
                sentence = " ".join(json.loads(sentence))
            if not case_sensitive:
                sentence = sentence.lower()
            sentences.append(sentence)
    return sentences

def extract_sentences_from_folder(folder: str, case_sensitive:bool):
    sentences = []  # type: List[str]
    for file in sorted(os.listdir(folder)):
        with open(os.path.join(folder, file)) as f:
            sentence = f.read()
            if not case_sensitive:
                sentence.lower()
            sentences.append(sentence)
    return sentences

def run(args):
    references_file = args['REFERENCES_FILE']
    predictions_file = args['PREDICTIONS_FILE']
    file_type = args['--format'] or 'jsonl'
    case_sensitive = args.get('--case_sensitive', False)

    if file_type != 'textfolder':
        references = extract_sentences(references_file, file_type, case_sensitive)
        predictions = extract_sentences(predictions_file, file_type, case_sensitive)
    elif file_type == 'textfolder':
        references = extract_sentences_from_folder(references_file, case_sensitive)
        predictions = extract_sentences_from_folder(predictions_file, case_sensitive)

    assert len(references) == len(predictions), 'References and predictions are not of the same length: reference: %s, predictions: %s' % (len(references), len(predictions))

    if not args['--use-rouge155']:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(hyps=predictions, refs=references, avg=True)
        print(scores)
    else:
        import pyrouge
        with tempfile.TemporaryDirectory() as data_dir:
            # First convert to single files
            ref_dir = os.path.join(data_dir, 'references')
            os.makedirs(ref_dir)

            dec_dir = os.path.join(data_dir, 'decoded')
            os.makedirs(dec_dir)

            for i, (decoded, reference) in enumerate(zip(predictions, references)):
                with open(os.path.join(ref_dir, '%06d_reference.txt' % i), 'w') as f:
                    f.write(reference.replace('.', '.\n'))
                with open(os.path.join(dec_dir, '%06d_decoded.txt' % i), 'w') as f:
                    f.write(decoded.replace('.', '.\n'))

            r = pyrouge.Rouge155()
            r.model_filename_pattern = '#ID#_reference.txt'
            r.system_filename_pattern = '(\d+)_decoded.txt'
            r.model_dir = ref_dir
            r.system_dir = dec_dir
            logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
            rouge_results = r.convert_and_evaluate()
            results_dict = r.output_to_dict(rouge_results)
            print(results_dict)
            print()
            log_str = ""
            for x in ["1","2","l"]:
                log_str += "\nROUGE-%s:\n" % x
                for y in ["f_score", "recall", "precision"]:
                    key = "rouge_%s_%s" % (x,y)
                    key_cb = key + "_cb"
                    key_ce = key + "_ce"
                    val = results_dict[key]
                    val_cb = results_dict[key_cb]
                    val_ce = results_dict[key_ce]
                    log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
            print(log_str)


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
