from nltk.translate.bleu_score import sentence_bleu
from wmt_data import *
import numpy as np

def bleu(gold, sent):
    try:
        return sentence_bleu([gold], sent)
    except:
        return 0.0

def buildSentence(ids, vocab):
    sent = np.array([vocab[i] for i in ids])
    if '_EOS' in sent:
        index = min([i for i,x in enumerate(sent) if x == '_EOS'])
        sent = sent[:index]
    return " ".join(sent).replace('_PAD', '').replace('_EOS', '')

def computeBleu(sess, model, dev_set, rev_vocab_en, rev_vocab_fr):
    bleus = []
    for bucket_id in xrange(len(model.buckets)):
        encoder_inputs, decoder_inputs, target_weights = get_all_bucket(dev_set, model.buckets, bucket_id)
        bucket_size = len(encoder_inputs[0])
        model.batch_size = bucket_size
        loss, outputs = model.test_step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id)
        english_sentences = np.array(encoder_inputs).T
        french_translation = np.argmax(np.array(outputs), axis=2).T
        french_gold = np.array(decoder_inputs).T
        for i in range(bucket_size):
            en = buildSentence(reversed(english_sentences[i]), rev_vocab_en)
            fr = buildSentence(french_translation[i], rev_vocab_fr)
            fr_gold = buildSentence(french_gold[i][1:], rev_vocab_fr)
            bleus.append(bleu(fr_gold, fr))
    return np.mean(np.array(bleus))