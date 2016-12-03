#################################################################################
# Copyright 2016 Pragaash Ponnusamy. All Rights Reserved.                       #
#                                                                               #
# Licensed under the Apache License, Version 2.0 (the "License");               #
# you may not use this file except in compliance with the License.              #
# You may obtain a copy of the License at                                       #
#                                                                               #
#     http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                               #
# Unless required by applicable law or agreed to in writing, software           #
# distributed under the License is distributed on an "AS IS" BASIS,             #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.      #
# See the License for the specific language governing permissions and           #
# limitations under the License.                                                #
#################################################################################

"""
Utility functions to download English-French data from WMT'15,
tokenize them and create a vocabulary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import re
from collections import defaultdict

_FORCE_STREAMING = False
try:
    import dask.bag as db
    from dask.diagnostics import ProgressBar
except ImportError:
    _FORCE_STREAMING = True
    
# Global Constants.
_EN          = 'en'
_FR          = 'fr'
_TOKENIZER   = re.compile(r"([a-zA-Z]'|[\w]+|[.,!?)(\"':;])", re.U)
_START_VOCAB = ['_PAD', '_GO', '_EOS', '_UNK']
_PAD_ID      = 0
_GO_ID       = 1
_EOS_ID      = 2
_UNK_ID      = 3


def en_sanitizer(sentence):
    """
    Basic sanitizer for English data.
    """
    return sentence.strip().lower()


def fr_sanitizer(sentence):
    """
    Basic sanitizer for French data.
    """
    return sentence.strip().lower().replace(u'\u2019', u"'")


def build_vocab(data_file, output_dir, size=50000, lang='en'):
    """
    Builds vocab of <size> from <data_file> and stores it in <output_dir>.
    """
    b = db.read_text(data_file).str.strip().str.lower()
    if lang is 'fr':
        b = b.str.replace(u'\u2019', u"'")
    b = b.map(lambda s: _TOKENIZER.findall(s)).concat().frequencies().topk(size - 4, lambda x: x[1]).pluck(0)
    a = db.from_sequence(_START_VOCAB)
    c = db.concat([a, b]).repartition(1)
    save_path = '%s/vocab%d.%s' % (output_dir, size, lang)
    with ProgressBar():
        c.to_textfiles([save_path])
    return save_path


def streaming_build_vocab(data_file, output_dir, size=50000, lang='en'):
    vocab = defaultdict(int)
    save_path = '%s/vocab%d.%s' % (output_dir, size, lang)
    sanitize = fr_sanitizer if lang is 'fr' else en_sanitizer
    with codecs.open(data_file, 'r', encoding='utf-8') as df:
        for line in df:
            words = _TOKENIZER.findall(sanitize(line))
            for word in words:
                vocab[word] += 1
    vocab = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    with codecs.open(save_path, 'w', encoding='utf-8') as vf:
        for token in vocab[:size]:
            vf.write(token + '\n')
    return save_path


def get_vocab(vocab_dir, size, lang='en'):
    """
    Retrieves the vocab of specified <size> and <lang> from <vocab_dir> as
    a dictionary and the reverse vocab as a list.
    """
    vocab_file = '%s/vocab%d.%s' % (vocab_dir, size, lang)
    with codecs.open(vocab_file, 'r', encoding='utf-8') as vf:
        tokens = vf.read().strip().split('\n')
    vocab = dict(zip(tokens, range(size)))
    return vocab, tokens


def sentence_to_token_ids(sentence, vocab):
    """
    Subroutine to convert a sentence to a space separated string of token ids.
    """
    return ' '.join(map(lambda w: str(vocab.get(w, _UNK_ID)), sentence))


def data_to_token_ids(data_file, vocab, output_dir, size=50000, lang='en'):
    """
    Converts the <data_file> to a data of token ids as given by the <vocab> and
    stores the tokenized data file in <output_dir>.
    """
    b = db.read_text(data_file).str.strip().str.lower()
    if lang is 'fr':
        b = b.str.replace(u'\u2019' u"'")
    b = b.map(lambda s: sentence_to_token_ids(_TOKENIZER.findall(s), vocab))
    file_name, _ = os.path.splitext(os.path.basename(data_file))
    save_path = '%s/%s.ids%d.%s' % (output_dir, file_name, size, lang)
    with ProgressBar():
        b.to_textfiles([save_path])
    return save_path


def streaming_data_to_token_ids(data_file, vocab, output_dir, size=50000, lang='en'):
    file_name, _ = os.path.splitext(os.path.basename(data_file))
    save_path = '%s/%s.ids%d.%s' % (output_dir, file_name, size, lang)
    sanitize = fr_sanitizer if lang is 'fr' else en_sanitizer
    with codecs.open(save_path, 'w', encoding='utf-8') as tk:
        with codecs.open(data_file, 'r', encoding='utf-8') as df:
            for line in df:
                sentence = _TOKENIZER.findall(sanitize(line))
                tk.write(sentence_to_token_ids(sentence, vocab) + '\n')
    return save_path


def prepare_wmt_data(data_dir_dict, en_vocabulary_size, fr_vocabulary_size, streaming=True):
    """
    Get WMT data into data_dir, create vocabularies and tokenize data.

    Args:
        data_dir_dict: directory in which the data sets will be stored.
        en_vocabulary_size: size of the English vocabulary to create and use.
        fr_vocabulary_size: size of the French vocabulary to create and use.
        streaming: Process data by streaming from disk.

    Returns:
        A tuple of 6 elements:
            (1) path to the token-ids for English training data-set,
            (2) path to the token-ids for French training data-set,
            (3) path to the English vocabulary file,
            (4) path to the French vocabulary file.
    """
    # Check Streaming Capability.
    streaming = True if not streaming and _FORCE_STREAMING else streaming
    
    # Validate Input Arguments.
    keys = ['data_dir', 'en_data', 'fr_data']
    key_error_msg = 'Expected ({}) in data_dir_dict.'.format(', '.join(keys))
    vocab_error_msg = 'Expected vocab size to be greater than 0.'
    assert all(map(data_dir_dict.__contains__, keys)), key_error_msg
    assert en_vocabulary_size > 0 and fr_vocabulary_size > 0, vocab_error_msg
    
    # Fix Functions.
    vocab_builder = streaming_build_vocab if streaming else build_vocab
    tokenizer = streaming_data_to_token_ids if streaming else data_to_token_ids

    # Extract keys from Data Dictionary.
    data_dir, en_data, fr_data = map(data_dir_dict.__getitem__, keys)
    data_dir = os.path.abspath(data_dir)

    # Create vocabularies of the appropriate sizes.
    en_path = os.path.join(data_dir, en_data)
    fr_path = os.path.join(data_dir, fr_data)
    print('Building English Vocab...')
    en_vocab_path = vocab_builder(en_path, data_dir, size=en_vocabulary_size, lang=_EN)
    print('Building French Vocab...')
    fr_vocab_path = vocab_builder(fr_path, data_dir, size=fr_vocabulary_size, lang=_FR)

    # Create token ids for the training data.
    en_vocab, _ = get_vocab(data_dir, en_vocabulary_size, lang=_EN)
    fr_vocab, _ = get_vocab(data_dir, fr_vocabulary_size, lang=_FR)
    print('Tokenizing English Data...')
    en_train_ids_path = tokenizer(en_path, en_vocab, data_dir, size=en_vocabulary_size, lang=_EN)
    print('Tokenizing French Data...')
    fr_train_ids_path = tokenizer(fr_path, fr_vocab, data_dir, size=fr_vocabulary_size, lang=_FR)

    return en_train_ids_path, fr_train_ids_path, en_vocab_path, fr_vocab_path
