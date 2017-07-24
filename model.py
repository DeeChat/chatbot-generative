from __future__ import print_function

import json
import os
import time

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import rnn

import config

# TODO: fix unknown token; add dropout; decay learning rate; apply beam search.


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                enc_cell,
                                dec_cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    """Embedding sequence-to-sequence model with attention.
    Re-implement of `tensorflow.contrib.legacy_seq2seq.embedding_attention_seq2seq`.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Warning: when output_projection is None, the size of the attention vectors
    and variables will be made proportional to num_decoder_symbols, can be large.

    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        enc_cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
        dec_cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_decoder_symbols] and B has
            shape [num_decoder_symbols]; if provided and feed_previous=True, each
            fed previous output will first be multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
            of decoder_inputs will be used (the "GO" symbol), and all other decoder
            inputs will be taken from previous outputs (as in embedding_rnn_decoder).
            If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype of the initial RNN state (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_seq2seq".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x num_decoder_symbols] containing the generated
                outputs.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with tf.variable_scope(scope or "embedding_attention_seq2seq",
                           dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        enc_cell = core_rnn_cell.EmbeddingWrapper(
            enc_cell,
            embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)

        # TODO: use `tensorflow.nn.bidirectional_dynamic_rnn` instead.
        encoder_outputs, encoder_state = rnn.static_rnn(
            enc_cell, encoder_inputs, dtype=dtype)
        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [
            array_ops.reshape(e, [-1, 1, enc_cell.output_size]) for e in encoder_outputs
        ]
        attention_states = array_ops.concat(top_states, 1)

        # Decoder.
        output_size = None
        if output_projection is None:
            dec_cell = core_rnn_cell.OutputProjectionWrapper(dec_cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            return tf.contrib.legacy_seq2seq.embedding_attention_decoder(
                decoder_inputs,
                encoder_state,
                attention_states,
                dec_cell,
                num_decoder_symbols,
                embedding_size,
                num_heads=num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=reuse):
                outputs, dec_state = tf.contrib.legacy_seq2seq.embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state,
                    attention_states,
                    dec_cell,
                    num_decoder_symbols,
                    embedding_size,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False,
                    initial_state_attention=initial_state_attention)
                dec_state_list = [dec_state]
                if tf.nest.is_sequence(dec_state):
                    dec_state_list = tf.nest.flatten(dec_state)
                return outputs + dec_state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if tf.nest.is_sequence(encoder_state):
            state = tf.nest.pack_sequence_as(
                structure=encoder_state, flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state


class ChatBotModel(object):
    def __init__(self, forward_only, batch_size):
        """
        :param forward_only: if set, we do not construct the backward pass in the model.
        """
        print("Initializing new model...")
        self.fw_only = forward_only
        self.batch_size = batch_size
        with open(os.path.join(config.DATA_PATH, "vocab_size.json"), "r") as f:
            self.vocab_size = json.load(f)

    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print("Creating placeholders...")
        self.encoder_inputs = [tf.placeholder(
            tf.int32, shape=[None], name='encoder{}'.format(i)
        ) for i in range(config.BUCKETS[-1][0])]
        # config.BUCKETS[-1][0]): max context length
        self.decoder_inputs = [tf.placeholder(
            tf.int32, shape=[None], name='decoder{}'.format(i)
        ) for i in range(config.BUCKETS[-1][1] + 1)]
        # config.BUCKETS[-1][1]): max utterance length
        self.decoder_masks = [tf.placeholder(
            tf.float32, shape=[None], name='mask{}'.format(i)
        ) for i in range(config.BUCKETS[-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = self.decoder_inputs[1:]

    def _inference(self):
        print('Creating inference...')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < config.NUM_SAMPLES < self.vocab_size['decoder']:
            w_t = tf.get_variable('proj_w', [self.vocab_size['decoder'], config.HIDDEN_SIZE])
            # `config.HIDDEN_SIZE` is the same as embedding size
            w = tf.transpose(w_t)
            b = tf.get_variable('proj_b', [self.vocab_size['decoder']])
            self.output_projection = (w, b)

        def sampled_loss(labels, logits):
            labels = tf.reshape(labels, [-1, 1])
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(logits, tf.float32)
            return tf.nn.sampled_softmax_loss(
                weights=local_w_t,
                biases=local_b,
                labels=labels,
                inputs=local_inputs,
                num_sampled=config.NUM_SAMPLES,
                num_classes=self.vocab_size['decoder'])

        self.softmax_loss_function = sampled_loss

        enc_cell = tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE)

        self.enc_cell = tf.contrib.rnn.MultiRNNCell([enc_cell] * config.NUM_LAYERS)
        dec_cell = tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE)
        self.dec_cell = tf.contrib.rnn.MultiRNNCell([dec_cell] * config.NUM_LAYERS)

    def _create_loss(self):
        print('Creating loss...')
        start = time.time()

        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, self.enc_cell, self.dec_cell,
                num_encoder_symbols=self.vocab_size['encoder'],
                num_decoder_symbols=self.vocab_size['decoder'],
                embedding_size=config.HIDDEN_SIZE,
                output_projection=self.output_projection,
                feed_previous=do_decode)

        if self.fw_only:
            # While testing.
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.decoder_masks,
                config.BUCKETS,
                lambda x, y: _seq2seq_f(x, y, True),
                softmax_loss_function=self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection:
                for bucket in range(len(config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(
                        output, self.output_projection[0]
                    ) + self.output_projection[1] for output in self.outputs[bucket]]
        else:
            # While training.
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.decoder_masks,  # weights
                config.BUCKETS,
                lambda x, y: _seq2seq_f(x, y, False),  # seq2seq
                softmax_loss_function=self.softmax_loss_function)
        print('Time:', time.time() - start)

    def _create_optimizer(self):
        print('Create optimizer... ')
        with tf.variable_scope('training'):
            self.global_step = tf.Variable(
                0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainable_vars = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket_id in range(len(config.BUCKETS)):
                    clipped_grads, norm = tf.clip_by_global_norm(
                        tf.gradients(self.losses[bucket_id], trainable_vars),
                        config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(
                        zip(clipped_grads, trainable_vars),
                        global_step=self.global_step))
                    print('Creating opt for bucket {:d} took {:.2f} seconds.'.format(
                        bucket_id, time.time() - start))
                    start = time.time()

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._create_optimizer()
