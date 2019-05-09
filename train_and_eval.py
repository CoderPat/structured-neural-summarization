import argparse

from opengnn.models import GraphToSequence, SequencedGraphToSequence

from opengnn.encoders import GGNNEncoder, SequencedGraphEncoder
from opengnn.decoders.sequence import RNNDecoder, HybridPointerDecoder
from opengnn.inputters import TokenEmbedder, CopyingTokenEmbedder
from opengnn.inputters import GraphEmbedder
from opengnn.inputters import SequencedGraphInputter
from opengnn.utils import CoverageBahdanauAttention, read_jsonl_gz_file

import tensorflow as tf
import os
import json

from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.python.util import function_utils
from tensorflow.python import debug as tf_debug

from rouge import Rouge

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


DEFAULT_TRAIN_SOURCE_FILE = 'data/naturallanguage/cnn_dailymail/split/train/inputs.jsonl.gz'
DEFAULT_TRAIN_TARGET_FILE = 'data/naturallanguage/cnn_dailymail/split/train/summaries.jsonl.gz'

DEFAULT_VALID_SOURCE_FILE = 'data/naturallanguage/cnn_dailymail/split/valid/inputs.jsonl.gz'
DEFAULT_VALID_TARGET_FILE = 'data/naturallanguage/cnn_dailymail/split/valid/summaries.jsonl.gz'

DEFAULT_NODE_VOCAB_FILE = 'data/naturallanguage/cnn_dailymail/node.vocab'
DEFAULT_EDGE_VOCAB_FILE = 'data/naturallanguage/cnn_dailymail/edge.vocab'
DEFAULT_TARGET_VOCAB_FILE = 'data/naturallanguage/cnn_dailymail/output.vocab'

DEFAULT_MODEL_NAME = 'cnndailymail_summarizer'


def main():
    # argument parsing
    parser = argparse.ArgumentParser()

    # optimization arguments
    parser.add_argument('--optimizer', default='adam', type=str,
                        help="Number of epochs to train the model")
    parser.add_argument('--train_steps', default=300000, type=int,
                        help="Number of steps to optimize")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help="The learning rate for the optimizer")
    parser.add_argument('--lr_decay_rate', default=0.0, type=float,
                        help="Learning rate decay rate")
    parser.add_argument('--lr_decay_steps', default=10000, type=float,
                        help="Number of steps between learning rate decay application")
    parser.add_argument('--adagrad_initial_accumulator', default=0.1, type=float,
                        help="Number of epochs to train the model")
    parser.add_argument('--momentum_value', default=0.95, type=float,
                        help="Number of epochs to train the model")
    parser.add_argument('--batch_size', default=16, type=int,
                        help="Number of epochs to train the model")
    parser.add_argument('--sample_buffer_size', default=10000, type=int,
                        help="The number of samples in the buffer shuffled before training")
    parser.add_argument('--bucket_width', default=5, type=int,
                        help="Range of allowed lengths in a batch. Optimizes RNN loops")
    parser.add_argument('--clip_gradients', default=5., type=float,
                        help="Maximum norm of the gradients")
    parser.add_argument('--validation_interval', default=20000, type=int,
                        help="The number of training steps between each validation run")
    parser.add_argument('--validation_metric', default='rouge', type=str,
                        help="The metric to compare models between validations")
    parser.add_argument('--patience', default=5, type=int,
                        help="Number of worse validations needed to trigger early stop")
    parser.add_argument('--logging_window', default=200, type=int,
                        help="Number of steps taken when logging")

    # model options arguments
    parser.add_argument('--source_embeddings_size', default=128, type=int,
                        help="Size of the input tokens embeddings")
    parser.add_argument('--target_embeddings_size', default=128, type=int,
                        help="Size of the target token embeddings")
    parser.add_argument('--embeddings_dropout', default=0.2, type=float,
                        help="Dropout applied to the node embeddings during training")
    parser.add_argument('--node_features_size', default=256, type=int,
                        help="Size of the node features hidden state")
    parser.add_argument('--node_features_dropout', default=0.2, type=float,
                        help="Dropout applied to the node features during training")
    parser.add_argument('--ggnn_num_layers', default=4, type=int,
                        help="Number of GGNN layers with distinct weights")
    parser.add_argument('--ggnn_timesteps_per_layer', default=1, type=int,
                        help="Number of GGNN propagations per layer")
    parser.add_argument('--rnn_num_layers', default=1, type=int,
                        help="Number of layers in the input and output rnns")
    parser.add_argument('--rnn_hidden_size', default=256, type=int,
                        help="Size of the input and output rnns hidden state")
    parser.add_argument('--rnn_hidden_dropout', default=0.3, type=float,
                        help="Dropout applied to the rnn hidden state during training")
    parser.add_argument('--attend_all_nodes', default=False, action='store_true',
                        help="If enabled, attention and copying will consider all nodes "
                             "rather than only the ones in the primary sequence")
    parser.add_argument('--only_graph_encoder', default=False, action='store_true',
                        help="If enabled, the model will ignore the sequence encoder, "
                             "using only the graph structure")
    parser.add_argument('--ignore_graph_encoder', default=False, action='store_true',
                        help="If enabled, the model ignore the graph encoder, using only "
                             "the primary sequence encoder")
    parser.add_argument('--copy_attention', default=False, action='store_true',
                        help="Number of epochs to train the model")
    parser.add_argument('--coverage_layer', default=False, action='store_true',
                        help="Number of epochs to train the model")
    parser.add_argument('--coverage_loss', default=0., type=float,
                        help="Number of epochs to train the model")
    parser.add_argument('--max_iterations', default=120, type=int,
                        help="The maximum number of decoding iterations at inference time")
    parser.add_argument('--beam_width', default=10, type=int,
                        help="The number of beam to search while decoding")
    parser.add_argument('--length_penalty', default=1.0, type=float,
                        help="The length ")
    parser.add_argument('--case_sensitive', default=False, action='store_true',
                        help="If enabled, node labels are case sentitive")

    # arguments for loading data
    parser.add_argument('--train_source_file', default=DEFAULT_TRAIN_SOURCE_FILE, type=str,
                        help="Path to the jsonl.gz file containing the train input graphs")
    parser.add_argument('--train_target_file', default=DEFAULT_TRAIN_TARGET_FILE, type=str,
                        help="Path to the jsonl.gz file containing the train input graphs")
    parser.add_argument('--valid_source_file', default=DEFAULT_VALID_SOURCE_FILE, type=str,
                        help="Path to the jsonl.gz file containing the valid input graphs")
    parser.add_argument('--valid_target_file', default=DEFAULT_VALID_TARGET_FILE, type=str,
                        help="Path to the jsonl.gz file containing the valid input graphs")
    parser.add_argument('--infer_source_file', default=None,
                        help="Path to the jsonl.gz file in which we wish to do inference "
                             "after training is complete")
    parser.add_argument('--infer_predictions_file', default=None,
                        help="Path to the file to save the results from inference")
    parser.add_argument('--node_vocab_file', default=DEFAULT_NODE_VOCAB_FILE, type=str,
                        help="Path to the json containing the dataset")
    parser.add_argument('--edge_vocab_file', default=DEFAULT_EDGE_VOCAB_FILE, type=str,
                        help="Path to the json containing the dataset")
    parser.add_argument('--target_vocab_file', default=DEFAULT_TARGET_VOCAB_FILE, type=str,
                        help="Path to the json containing the dataset")
    parser.add_argument('--truncated_source_size', default=500, type=int,
                        help="Max size for source sequences in the input graphs after truncation")
    parser.add_argument('--truncated_target_size', default=100, type=int,
                        help="Max size for target sequences after truncation")
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME, type=str,
                        help="Model name")

    # arguments for persistence
    parser.add_argument('--checkpoint_interval', default=5000, type=int,
                        help="The number of steps between model checkpoints")
    parser.add_argument('--checkpoint_dir', default=None, type=str,
                        help="Directory to where to save the checkpoints")

    # arguments for debugging
    parser.add_argument('--debug_mode', default=False, action='store_true',
                        help="If true, it will enable the tensorflow debugger")

    args = parser.parse_args()

    model = build_model(args)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.model_name

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        os.makedirs(os.path.join(args.checkpoint_dir, "valid"))
    elif not os.path.exists(os.path.join(args.checkpoint_dir, "valid")):
        os.makedirs(os.path.join(args.checkpoint_dir, "valid"))

    train_and_eval(model, args)

    if args.infer_source_file is not None:
        infer(model, args)


def train_and_eval(model, args):
    optimizer = build_optimizer(args)
    metadata = build_metadata(args)
    config = build_config(args)
    params = build_params(args)

    train_input_fn = model.input_fn(
        mode=tf.estimator.ModeKeys.TRAIN,
        batch_size=args.batch_size,
        metadata=metadata,
        features_file=args.train_source_file,
        labels_file=args.train_target_file,
        features_bucket_width=args.bucket_width,
        sample_buffer_size=args.sample_buffer_size)
    valid_input_fn = model.input_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=args.batch_size,
        metadata=metadata,
        features_file=args.valid_source_file,
        labels_file=args.valid_target_file,)
    valid_targets = read_jsonl_gz_file(args.valid_target_file)

    train_iterator = get_iterator_from_input_fn(train_input_fn)
    valid_iterator = get_iterator_from_input_fn(valid_input_fn)
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=False))
    with tf.Session(config=session_config) as session:
        if args.debug_mode:
            session = tf_debug.LocalCLIDebugWrapperSession(
                session, dump_root="~/Downloads/tf-debug")

        # build train graph, loss and optimization ops
        features, labels = train_iterator.get_next()
        with tf.variable_scope(args.model_name):
            outputs, _ = model(
                features, labels, tf.estimator.ModeKeys.TRAIN, params, config)
            train_loss, train_tb_loss = model.compute_loss(
                features, labels, outputs, params, tf.estimator.ModeKeys.TRAIN)

        train_op = optimizer(train_loss)

        # build eval graph, loss and prediction ops
        features, labels = valid_iterator.get_next()
        with tf.variable_scope(args.model_name, reuse=True):
            outputs, predictions = model(
                features, labels, tf.estimator.ModeKeys.EVAL, params, config)
            _, valid_tb_loss = model.compute_loss(
                features, labels, outputs, params, tf.estimator.ModeKeys.EVAL)

        global_step = tf.train.get_global_step()

        best_metric = 0
        worse_epochs = 0

        saver = tf.train.Saver(max_to_keep=100)
        train_summary = Summary(args.checkpoint_dir)
        valid_summary = Summary(os.path.join(args.checkpoint_dir, "valid"))
        # TODO: Initialize tables some other way
        session.run([
            train_iterator.initializer,
            tf.tables_initializer()])

        # check if we are restarting a run
        latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint is not None:
            saver.restore(session, latest_checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        initial_step = session.run(global_step)

        window_loss = 0
        window_steps = 0
        for train_step in range(initial_step+1, args.train_steps+1):
            step_loss, _ = session.run([train_tb_loss, train_op])
            window_loss += step_loss
            window_steps += 1

            # check if in logging schedule
            if train_step % args.logging_window == 0:
                train_summary.scalar("loss", window_loss / window_steps, train_step)
                print("step %d, train loss: %0.2f" %
                      (train_step, window_loss / window_steps))
                window_loss = 0
                window_steps = 0

            # and checkpointing schedule
            if train_step % args.checkpoint_interval == 0:
                print("saving current model...")
                saver.save(session, os.path.join(args.checkpoint_dir, "current.ckpt"), global_step)

            # after training, do evaluation if on schedule
            if train_step % args.validation_interval == 0:
                valid_loss, valid_rouge = evaluate(
                    session,
                    model,
                    valid_iterator,
                    valid_tb_loss,
                    predictions,
                    valid_targets)
                print("eval loss: %0.2f, eval rouge: %0.2f" % (valid_loss, valid_rouge))
                valid_summary.scalar("loss", valid_loss, train_step)
                valid_summary.scalar("rouge", valid_rouge, train_step)
                if args.validation_metric == "rouge":
                    # check for new best model
                    if valid_rouge > best_metric:
                        best_metric = valid_rouge
                        worse_epochs = 0
                        print("saving best model...")
                        saver.save(session, os.path.join(args.checkpoint_dir, "best.ckpt"))
                    else:
                        worse_epochs += 1

                    # and stop training if triggered patience
                    if worse_epochs >= args.patience:
                        print("early stopping triggered...")
                        break
                else:
                    raise ValueError("%s not supported as validation metric" %
                                     args.validation_metric)


def evaluate(session,
             model,
             iterator,
             loss,
             predictions,
             targets):
    """ """
    valid_loss = 0
    valid_steps = 0

    valid_predictions = []

    session.run([iterator.initializer, tf.tables_initializer()])
    while True:
        try:
            batch_loss, batch_predictions = session.run([loss, predictions])
            batch_predictions = [model.process_prediction({"tokens": prediction})
                                 for prediction in batch_predictions["tokens"]]

            valid_loss += batch_loss
            valid_predictions = valid_predictions + batch_predictions
            valid_steps += 1
        except tf.errors.OutOfRangeError:
            break

    loss = valid_loss / valid_steps
    rouge = compute_rouge(valid_predictions, targets)
    return loss, rouge


def get_iterator_from_input_fn(input_fn):
    with tf.device('/cpu:0'):
        return input_fn().make_initializable_iterator()


def build_model(args):
    """"""
    if args.coverage_layer:
        attention_layer = CoverageBahdanauAttention
    else:
        attention_layer = BahdanauAttention

    if args.copy_attention:
        node_embedder = CopyingTokenEmbedder(
            vocabulary_file_key="node_vocabulary",
            output_vocabulary_file_key="target_vocabulary",
            embedding_size=args.source_embeddings_size,
            dropout_rate=args.embeddings_dropout,
            lowercase=not args.case_sensitive)
        target_inputter = CopyingTokenEmbedder(
            vocabulary_file_key="target_vocabulary",
            input_tokens_fn=lambda data: data['labels'],
            embedding_size=args.target_embeddings_size,
            dropout_rate=args.embeddings_dropout,
            truncated_sentence_size=args.truncated_target_size)
        decoder = HybridPointerDecoder(
            num_units=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            output_dropout_rate=args.rnn_hidden_dropout,
            attention_mechanism_fn=attention_layer,
            coverage_loss_lambda=args.coverage_loss,
            copy_state=True)
    else:
        node_embedder = TokenEmbedder(
            vocabulary_file_key="node_vocabulary",
            embedding_size=args.source_embeddings_size,
            dropout_rate=args.embeddings_dropout,
            lowercase=not args.case_sensitive)
        target_inputter = TokenEmbedder(
            vocabulary_file_key="target_vocabulary",
            embedding_size=args.target_embeddings_size,
            dropout_rate=args.embeddings_dropout,
            truncated_sentence_size=args.truncated_target_size)
        decoder = RNNDecoder(
            num_units=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            output_dropout_rate=args.rnn_hidden_dropout,
            attention_mechanism_fn=attention_layer,
            coverage_loss_lambda=args.coverage_loss,
            copy_state=True)

    if args.only_graph_encoder:
        model = GraphToSequence(
            source_inputter=GraphEmbedder(
                edge_vocabulary_file_key="edge_vocabulary",
                node_embedder=node_embedder),
            target_inputter=target_inputter,
            encoder=GGNNEncoder(
                num_timesteps=[args.ggnn_timesteps_per_layer
                               for _ in range(args.ggnn_num_layers)],
                node_feature_size=args.node_features_size,
                gru_dropout_rate=args.node_features_dropout),
            decoder=decoder,
            name=args.model_name)
    else:
        model = SequencedGraphToSequence(
            source_inputter=SequencedGraphInputter(
                graph_inputter=GraphEmbedder(
                    edge_vocabulary_file_key="edge_vocabulary",
                    node_embedder=node_embedder),
                truncated_sequence_size=args.truncated_source_size),
            target_inputter=target_inputter,
            encoder=SequencedGraphEncoder(
                base_graph_encoder=GGNNEncoder(
                    num_timesteps=[args.ggnn_timesteps_per_layer
                                   for _ in range(args.ggnn_num_layers)],
                    node_feature_size=args.node_features_size,
                    gru_dropout_rate=args.node_features_dropout),
                gnn_input_size=args.node_features_size,
                encoder_type='bidirectional_rnn',
                num_units=args.rnn_hidden_size,
                num_layers=args.rnn_num_layers,
                dropout_rate=args.rnn_hidden_dropout,
                ignore_graph_encoder=args.ignore_graph_encoder,),
            decoder=decoder,
            only_attend_primary=not args.attend_all_nodes,
            name=args.model_name)
    return model


def infer(model, args):
    metadata = build_metadata(args)
    config = build_config(args)
    params = build_params(args)
    input_fn = model.input_fn(
        mode=tf.estimator.ModeKeys.PREDICT,
        batch_size=args.batch_size,
        metadata=metadata,
        features_file=args.infer_source_file)
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=False))

    iterator = get_iterator_from_input_fn(input_fn)
    with tf.Session(config=session_config) as session:
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(session, os.path.join(args.checkpoint_dir, "best.ckpt"))

        # build eval graph, loss and prediction ops
        features = iterator.get_next()
        with tf.variable_scope(args.model_name, reuse=True):
            _, predictions = model(
                features, None, tf.estimator.ModeKeys.PREDICT, params, config)

        session.run([iterator.initializer, tf.tables_initializer()])

        steps = 0
        infer_predictions = []
        while True:
            try:
                batch_predictions = session.run(predictions)
                batch_predictions = [model.process_prediction({"tokens": prediction})
                                     for prediction in batch_predictions["tokens"]]

                infer_predictions = infer_predictions + batch_predictions
                steps += 1
            except tf.errors.OutOfRangeError:
                break

    with open(args.infer_predictions_file, 'w') as out_file:
        for prediction in infer_predictions:
            out_file.write(json.dumps(prediction) + "\n")


def build_metadata(args):
    metadata = {
        "node_vocabulary": args.node_vocab_file,
        "edge_vocabulary": args.edge_vocab_file,
        "target_vocabulary": args.target_vocab_file
    }
    return metadata


def build_config(args):
    config = {
        # TODO
    }
    return config


def build_params(args):
    params = {
        'maximum_iterations': args.max_iterations,
        'beam_width': args.beam_width,
        'length_penalty': args.length_penalty
    }
    return params


def build_optimizer(args):
    global_step = tf.train.get_or_create_global_step()

    optimizer = args.optimizer
    if optimizer == 'adam':
        optimizer_class = tf.train.AdamOptimizer
        kwargs = {}
    elif optimizer == "adagrad":
        optimizer_class = tf.train.AdagradOptimizer
        kwargs = {"initial_accumulator_value": args.adagrad_initial_accumulator}
    elif optimizer == "momentum":
        optimizer_class = tf.train.MomentumOptimizer
        kwargs = {"momentum": args.momentum_value, "use_nesterov": True}
    else:
        optimizer_class = getattr(tf.train, optimizer, None)
        if optimizer_class is None:
            raise ValueError("Unsupported optimizer %s" % optimizer)
        kwargs = {}
        # TODO: optimizer params
        # optimizer_params = params.get("optimizer_params", {})

    def optimizer(lr): return optimizer_class(lr, **kwargs)

    learning_rate = args.learning_rate
    if args.lr_decay_rate:
        learning_rate = tf.train.exponential_decay(
            learning_rate,
            global_step,
            decay_steps=args.lr_decay_steps,
            decay_rate=args.lr_decay_rate,
            staircase=True)

    return lambda loss: tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        clip_gradients=args.clip_gradients,
        summaries=[
            "learning_rate",
            "global_gradient_norm",
        ],
        optimizer=optimizer,
        name="optimizer")


def compute_rouge(predictions, targets):
    predictions = [" ".join(prediction).lower() for prediction in predictions]
    predictions = [prediction if prediction else "EMPTY" for prediction in predictions]
    targets = [" ".join(target).lower() for target in targets]
    targets = [target if target else "EMPTY" for target in targets]
    rouge = Rouge()
    scores = rouge.get_scores(hyps=predictions, refs=targets, avg=True)
    return scores['rouge-2']['f']


class Summary(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, model_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(model_dir)

    def scalar(self, tag, value, step, family=None):
        """Log a scalar variable.

        Parameter
        ----------
        tag: basestring
            Name of the scalar
        value
        step: int
            training iteration
        """
        tag = os.path.join(family, tag) if family is not None else tag
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


if __name__ == "__main__":
    main()
