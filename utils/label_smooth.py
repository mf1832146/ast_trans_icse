import torch
import torch.nn as nn
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    def __init__(self, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, x, target):
        x = x.contiguous().view(-1, x.size(-1))

        ntokens = (target != 0).data.sum()
        target = target.contiguous().view(-1)
        vocab_size = x.size(1)

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.sum() > 0 and len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        loss = self.criterion(x, Variable(true_dist, requires_grad=False))
        return loss / ntokens