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

def computeBleu(sess, model, dev_set, rev_vocab_en, rev_vocab_fr, ppb=0):
    bleus = []
    for b_id in xrange(len(model.buckets)):
        inp, gold, _ = get_all_bucket(dev_set, model.buckets, b_id)
        outputs = model.test_step(sess, inp, b_id)
        english_sentences = np.array(inp).T
        french_translation = np.argmax(np.array(outputs), axis=2).T
        french_gold = np.array(gold).T
        for i in range(len(inp[0])):
            en = buildSentence(reversed(english_sentences[i]), rev_vocab_en)
            fr = buildSentence(french_translation[i], rev_vocab_fr)
            fr_gold = buildSentence(french_gold[i][1:], rev_vocab_fr)
            if i < ppb:
                print en, "\n", fr, "\n", fr_gold
                print "------------------"
            bleus.append(bleu(fr_gold, fr))
    return np.mean(np.array(bleus))
