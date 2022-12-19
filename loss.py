import torch
import torch.nn as nn

class LSEP(nn.Module):
    def __init__(self):
        super(LSEP, self).__init__()

    def forward(self, T, bayes, partial):
        # T: 128*10*10, bayes: 128, partial: 128*10
        T_transpose = T.transpose(1, 2)
        bayes_one_hot = torch.zeros(bayes.size(0), 1, 10).to(bayes.device)
        bayes_one_hot.scatter_(2, bayes.view(bayes.size(0), 1, 1), 1)
        q = torch.matmul(T_transpose, bayes_one_hot.transpose(1, 2)).view(bayes.size(0), 10)
        loss = 0.
        for i in range(bayes.size(0)):
            positive = torch.masked_select(q[i, :], torch.eq(partial[i, :], 1))
            negative = torch.masked_select(q[i, :], torch.eq(partial[i, :], 0))

            exp_sub = torch.exp(negative[:, None] - positive[None, :])
            exp_sum = torch.sum(exp_sub)

            loss += torch.log(1 + exp_sum)

        loss /= bayes.size(0)
        return loss
