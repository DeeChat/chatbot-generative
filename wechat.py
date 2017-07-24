from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from io import open

import itchat
import tensorflow as tf
from itchat.content import TEXT, MAP, CARD, NOTE, SHARING

import config
import data_utils
from chatbot import check_restore_parameters
from chatbot import construct_response
from chatbot import find_right_bucket
from chatbot import run_step
from model import ChatBotModel

sess = tf.InteractiveSession()
_, enc_vocab = data_utils.load_vocab(os.path.join(config.DATA_PATH, "vocab.enc"))
# `inv_dec_vocab` <type "list">: id2word.
inv_dec_vocab, __ = data_utils.load_vocab(os.path.join(config.DATA_PATH, "vocab.dec"))
model = ChatBotModel(True, batch_size=1)
model.build_graph()
saver = tf.train.Saver()
check_restore_parameters(sess, saver)
# Decode from standard input.
max_length = config.BUCKETS[-1][0]
print("Wechat server started. Say something. Max length is", max_length)


@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING],
                     isFriendChat=True, isGroupChat=False, isMpChat=False)
def wechat(msg):
    """
    Args:
        msg: can be TEXT, MAP, CARD NOTE and SHARING of wechat.
    Return:
        wechat response.
    """
    line = msg["Text"]
    token_ids = data_utils.sentence2id(enc_vocab, line)
    if len(token_ids) > max_length:
        print("Max length I can handle is:", max_length)
        # line = _get_user_input()
    # Which bucket does it belong to?
    bucket_id = find_right_bucket(len(token_ids))
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, decoder_masks = data_utils.get_batch(
        [(token_ids, [])], bucket_id, batch_size=1)
    # Get output logits for the sentence.
    _, _, output_logits = run_step(sess, model, encoder_inputs,
                                   decoder_inputs, decoder_masks,
                                   bucket_id, True)
    response = construct_response(output_logits, inv_dec_vocab)
    write_wechat_records(line, response)
    return response


# write chat records to file
def write_wechat_records(query, response):
    output_file = open(os.path.join(config.DATA_PATH, config.WECHAT_OUTPUT),
                       "a+", encoding="utf-8")
    output_file.write("HUMAN ++++ " + query + "\n")
    output_file.write("BOT ++++ " + response + "\n")


itchat.auto_login(hotReload=True)
itchat.run()
