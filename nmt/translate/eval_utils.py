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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from translate import data_utils
from nltk.translate.bleu_score import corpus_bleu


def _score(references, hypotheses):
    """
    Internal method to score reference corpus against the true corpus.
    """
    
    err_msg = 'Expected reference sentence to have non-zero length.'
    assert all(map(lambda x: len(x) > 0, references)), err_msg
    return corpus_bleu(references, hypotheses)

def _validate(x):
    """
    Internal method to check if the token, <x> is a valid word or
    if not, must be a _PAD symbol.
    """
    
    if x != data_utils._PAD_ID and x < data_utils._UNK_ID:
        raise StopIteration
    return x

def _reconstruct(sequence, reverse_vocab, sanitize=False):
    sequence = list(_validate(x) for x in sequence) if sanitize else sequence
    sentence = ' '.join([reverse_vocab[i] for i in sequence])
    sentence = sentence.replace(
        data_utils._START_VOCAB[data_utils._PAD_ID],
        '') if sanitize else sentence
    return sentence

def _build_data_matrix(data, bucket):
    """
    Generate batch matrix from bucket data.
    """
    
    enc_size, dec_size = bucket
    batch_size = len(data)
    
    batch_enc = np.zeros((enc_size, batch_size), dtype=np.int32)
    batch_dec = np.zeros((dec_size, batch_size), dtype=np.int32)
    batch_weights = np.ones((dec_size, batch_size), dtype=np.float32)
    batch_dec[0, :] = data_utils._GO_ID
    
    for i, (enc, dec) in enumerate(data):
        batch_enc[-len(enc):, i] = list(reversed(enc))
    
    return batch_enc, batch_dec, batch_weights

def compute_corpus_bleu_score(session, model, bucketed_data, buckets, reverse_vocab):
    """
    Compute the bleu score for the translation of the corpus, <bucketed_data>
    with the given <model>.
    """
    
    hypotheses, references = [], []
    for i, bucket in enumerate(buckets):
        data = bucketed_data[i]
        enc_in, dec_in, tgt_wt = _build_data_matrix(data, bucket)
        outputs = model.step(session, enc_in, dec_in, tgt_wt, i, forward_only=True)
        hypotheses.extend(map(lambda (x, y): _reconstruct(y, reverse_vocab), data))
        references.extend(map(lambda x: _reconstruct(x, reverse_vocab, True), outputs.T))
    return _score(references, hypotheses)

def compute_multi_corpus_bleu_score(session, model, multi_data, buckets, reverse_vocab):
    """
    Batch computation of the corpus bleu score for multiple corpi and returns
    the average score across the corpi.
    """
    
    scores = np.zeros(len(multi_data), dtype=np.float32)
    for i, data in enumerate(multi_data):
        scores[i] = compute_corpus_bleu_score(session, model, data, buckets, reverse_vocab)
    return scores.mean()
