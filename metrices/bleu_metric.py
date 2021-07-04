from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
import numpy as np

from metrices.bleu import compute_bleu


def batch_bleu(comments, predicts, nl_i2w, extra_vocab,  is_print):
    references, hypothesises = batch_evaluate(comments, predicts, nl_i2w, extra_vocab)
    scores = []
    for i in range(len(references)):
        if i < 10 and is_print:
            print('reference:', references[i])
            print('hypothesises:', hypothesises[i])
        bleu_score = compute_bleu([[references[i]]], [hypothesises[i]], smooth=True)[0]
        scores.append(bleu_score)
    return scores


def batch_evaluate(comments, predicts, nl_i2w, extra_vocab):
    batch_size = comments.size(0)
    references = []
    hypothesises = []
    for i in range(batch_size):
        reference = [nl_i2w[c.item()] if c.item() in nl_i2w else extra_vocab[c.item()] for c in comments[i]]
        if '</s>' in reference:
            reference = reference[:reference.index('</s>')]
        hypothesis = [nl_i2w[c.item()] if c.item() in nl_i2w else extra_vocab[c.item()] for c in predicts[i]]
        if '</s>' in hypothesis:
            hypothesis = hypothesis[:hypothesis.index('</s>')]
        if len(hypothesis) == 0:
            hypothesis = ['<?>']
        if len(reference) == 0:
            continue
        references.append(reference)
        hypothesises.append(hypothesis)
    return references, hypothesises


class BLEU4(Metric):
    def __init__(self, id2nl, output_transform=lambda x: x, device=None):
        super(BLEU4, self).__init__(output_transform, device=device)
        self._id2nl = id2nl
        self.is_print = True

    @reinit__is_reduced
    def reset(self):
        self._bleu_scores = 0
        self._num_examples = 0
        self.is_print = True

    @reinit__is_reduced
    def update(self, output):
        (y_pred, extra_vocab), y = output
        extra_vocab = {v.item(): k for k, v in extra_vocab.items()}
        scores = batch_bleu(y, y_pred, self._id2nl, extra_vocab, self.is_print)
        self._bleu_scores += np.sum(scores)
        self._num_examples += len(scores)
        self.is_print = False

    @sync_all_reduce("_bleu_scores", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("BLEU4 must have "
                                     "at least one example before it can be computed.")
        return self._bleu_scores / self._num_examples


