import torch


class CCELoss(torch.nn.Module):
    def __init__(self, b_lambda=1.0, reduction='mean'):
        super().__init__()
        self.b_lambda = b_lambda
        self.reduction = reduction

    def forward(self, b_x, b_y, b_blance):
        b_x = b_x[:, 0]
        b_blance[0] *= 1.1
        weights = torch.zeros(b_y.size(), dtype=b_y.dtype, device=b_y.device)
        weights += (b_y == 1).float() * b_blance[1]
        weights += (b_y == 0).float() * (b_blance[0]) * self.b_lambda
        edge_loss = torch.nn.BCEWithLogitsLoss(weights, reduction=self.reduction)(b_x, b_y)
        if self.reduction == 'sum':
            edge_loss /= b_y.size(0)
        return edge_loss
