import torch


class BaseOriLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.type_loss = ''

    def forward(self, ori_x, ori_y, edge_y):
        pass

    def getOri(self, ori_x):
        return ori_x
        pass

    def getOriXNnm(self):
        # orix 需要的channel数量
        return 1


class OORLoss(BaseOriLoss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, ori_x, ori_y, b_y):
        # b_mask = b_y == 1
        # ori_y = ori_y[b_mask]
        ori_x_sin, ori_x_cos = ori_x[:, 0], ori_x[:, 1]

        den = (ori_x_sin ** 2 + ori_x_cos ** 2) ** 0.5
        sin_x = ori_x_sin / den
        cos_x = ori_x_cos / den

        ori_f = (cos_x - torch.cos(ori_y)) ** 2 + \
            (sin_x - torch.sin(ori_y)) ** 2
        flags = torch.abs(ori_f) < 1
        ori_loss = (flags == 1).float() * 0.5 * (ori_f ** 2) + \
            (flags == 0).float() * (torch.abs(ori_f) - 0.5)
        if self.reduction == 'mean':
            ori_loss = torch.mean(ori_loss * b_y)
        else:  # self.reduction == 'sum':
            ori_loss = torch.sum(ori_loss * b_y)
            ori_loss /= b_y.size(0)
        return ori_loss

    def getOri(self, ori_x):
        return torch.atan2(ori_x[:, 0], ori_x[:, 1]).unsqueeze(1)

    def getOriXNnm(self):
        # orix 需要的channel数量
        return 2
