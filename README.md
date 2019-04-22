# Structured Neural Summarization

A repository with the code for [the paper](https://arxiv.org/abs/1811.01824) with the same title. The experiments are based on the more general-purpose graph neural network library [OpenGNN](https://github.com/CoderPat/OpenGNN). You can install it by following it's README.md.

Experiments are based around the `train_and_eval.py` script. Besides the main experiments, this repo also contains the following folders:

* [Parsers](parsers): A collection of scripts to parse and process various datasets to the format used by the experiments
* [Data](data): A collection of scripts to utility functions to handle and analyse the formated data
* [Models](models): Some bash script wrapppers around the main script with some model/hyperparameter combination for diferent experiments

## Getting Started

As an example, we will show how run a sequenced-graph to sequence model with attention on the [CNN/DailyMail](https://cs.nyu.edu/~kcho/DMQA/) dataset.
This assumed the process data is located in 
```/data/naturallanguage/cnn_dailymail/split/{train,valid,test}/{inputs,targets}.jsonl.gz```.

For instruction on how to process see the corresponding [subfolder](parsers/naturallanguage/dmcnn).

Start by build vocabularies for the node and edge labels in the input side and tokens in the output side by running

```bash
ognn-build-vocab --field_name node_labels \
                 --save_vocab /data/naturallanguage/cnn_dailymail/node.vocab \
                 /data/naturallanguage/cnn_dailymail/split/train/inputs.jsonl.gz
ognn-build-vocab --no_pad_token --field_name edges --string_index 0 \
                 --save_vocab /data/naturallanguage/cnn_dailymail/edge.vocab \
                 /data/naturallanguage/cnn_dailymail/split/train/inputs.jsonl.gz
ognn-build-vocab --with_sequence_tokens \
                 --save_vocab /data/naturallanguage/cnn_dailymail/output.vocab \
                 /data/naturallanguage/cnn_dailymail/split/train/targets.jsonl.gz 
```

Then run

```bash
python train_and_eval.py   
```

This will create the model directory `cnndailymail_summarizer`, which contains tensorflow checkpoint and event files that can monitored in tensorboard.

We can also pass directly the file we wish to do inference on by running

```bash
python train_and_eval.py --infer_source_file /data/naturallanguage/cnn_dailymail/split/test/inputs.jsonl.gz \
                         --infer_predictions_file /data/naturallanguage/cnn_dailymail/split/test/predictions.jsonl
```

Then print the metrics on the predictions run

```bash
python rouge_evaluator /data/naturallanguage/cnn_dailymail/split/test/summaries.jsonl.gz \
                       /data/naturallanguage/cnn_dailymail/split/test/predictions.jsonl
                         
```
