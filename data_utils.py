from __future__ import print_function
from __future__ import unicode_literals
import os
import random
import re
import json
from io import open
import jieba
import numpy as np
import config
jieba.load_userdict("dicts/user_dict.txt")


def make_dir(path):
    """
    Create a directory if there isn't one already.
    """
    if not os.path.exists(path):
        os.mkdir(path)


def basic_tokenizer(line, normalize_digits=True):
    """
    A tokenizer to tokenize text into tokens.
    :param line: a string of sentence.
    :param normalize_digits: whether to normalize digits.
    """
    words = []
    _digit_re = re.compile(r"\d")
    for token in jieba.cut(line.strip()):
        if token in [" ", ""]:
            continue
        if normalize_digits:
            token = re.sub(_digit_re, "#", token)
        words.append(token)
    return words


def build_vocab(filename):
    """
    Build vocabulary.
    :param filename: "train.enc" or "train.dec"
    """
    in_path = os.path.join(config.DATA_PATH, filename)
    out_path = os.path.join(config.DATA_PATH, "vocab.{}".format(filename[-3:]))
    # Word frequency statistics
    # `vocab`: dict of <token, frequency> pairs.
    vocab = {}
    with open(in_path, encoding="utf-8") as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
    # sort by frequency
    # sorted_vocab <type "list">
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("<pad>" + "\n")
        f.write("<unk>" + "\n")
        f.write("<s>" + "\n")
        f.write("<\s>" + "\n")
        index = 4
        for word in sorted_vocab:
            # Words with frequency less than `config.THRESHOLD`
            # should be dropped.
            if vocab[word] < config.THRESHOLD:
                return index
            f.write(word + "\n")
            index += 1
    return index


def load_vocab(vocab_path):
    """
    Load vocabulary.
    Returns:
        `words` is a list of vocab strings.
        `word2id` is a dict of <word_str, word_id> pairs.
    """
    with open(vocab_path, encoding="utf-8") as f:
        words = f.read().splitlines()
    word2id = {words[i]: i for i in range(len(words))}
    return words, word2id


def sentence2id(vocab, line):
    """
    Convert a sentence string to word id list.
    :param vocab: word2id
    :type vocab: dict
    :param line: a raw sentence
    :return: list of word indices
    """
    # Get word index or get the index of <unk>.
    return [vocab.get(token, vocab["<unk>"]) for token in basic_tokenizer(line)]


def token2id(data, mode):
    """
    Convert all the tokens in the data into their corresponding
        index in the vocabulary.
    Args:
        :param data: "train" or "test"
        :param mode: "enc" or "dec"
    """
    vocab_path = "vocab." + mode
    in_path = data + "." + mode
    out_path = data + "_ids." + mode
    # Get the dict of word2id: `vocab`.
    _, vocab = load_vocab(os.path.join(config.DATA_PATH, vocab_path))
    in_file = open(os.path.join(config.DATA_PATH, in_path),
                   encoding="utf-8")
    out_file = open(os.path.join(config.DATA_PATH, out_path), "w")
    lines = in_file.read().splitlines()
    # `lines` is a list of sentence strings, e.g., ["hello!", "how are you?"]
    in_file.close()
    for line in lines:
        if mode == "dec":  # we only care about "<s>" and </s> in encoder
            ids = [vocab["<s>"]]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab["<unk>"]) for token in basic_tokenizer(line)])
        if mode == "dec":
            ids.append(vocab["<\s>"])
        out_file.write(" ".join(str(id_) for id_ in ids) + "\n")
    out_file.close()


def process_data():
    print("Preparing data...")
    enc_vocab_size = build_vocab("train.enc")
    dec_vocab_size = build_vocab("train.dec")
    vocab_size = {"encoder": enc_vocab_size, "decoder": dec_vocab_size}
    with open(os.path.join(config.DATA_PATH, "vocab_size.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab_size, ensure_ascii=False))
    token2id("train", "enc")
    token2id("train", "dec")
    token2id("test", "enc")
    token2id("test", "dec")


def load_data(enc_filename, dec_filename):
    """
    Load data from *_ids.* files and group the data into buckets.
    Args:
        :param enc_filename: "train_ids.enc", etc.
        :param dec_filename: "train_ids.dec", etc.
    Return:
        `data_buckets` is a list of lists. Each list is a bucket,
        and each bucket contains many <context, response> pairs.
    """
    encode_file = open(os.path.join(config.DATA_PATH, enc_filename))
    decode_file = open(os.path.join(config.DATA_PATH, dec_filename))
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i + 1)
        # covert digit string to integer
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        # Pairs with too long context or utterance should be dropped.
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets


def _pad_input(input_, size):
    """
    Function for zero-padding.
    """
    return input_ + [config.PAD_ID] * (size - len(input_))


def _reshape_batch(inputs, size, batch_size):
    """
    Create batch-major inputs. Batch inputs are just re-indexed inputs.
    :param inputs: encoder_inputs or decoder_inputs, which is a list of batches.
    :param size: encoder_size or decoder_size.
    :param batch_size: batch size <type "int">.
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                      for batch_id in range(batch_size)],
                                     dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """
    :param data_bucket: a certain bucket from `data_buckets`,
        which is a list of <context, response> pairs.
    :param bucket_id: a bucket index which is randomly chosen.
    :param batch_size: batch size <type "int">.
    Return one batch to feed into the model.
    """
    # Only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # Pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))
    # encoder_inputs <type "list">: input data of encoder.
    # encoder_inputs[0]: first batch of encoder inputs.
    # encoder_inputs[0][0]: first context of the first batch.
    # encoder_inputs[0][0][0]: first word index of the context.
    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            # noinspection PyUnboundLocalVariable
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks


if __name__ == "__main__":
    process_data()
