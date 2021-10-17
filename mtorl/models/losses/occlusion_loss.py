import torch
from mtorl.models.losses.cceloss import CCELoss
from mtorl.models.losses.oorloss import OORLoss


class OcclousionLoss(torch.nn.Module):

    def __init__(self,
                 boundary_weights=[0.5, 0.5, 0.5, 0.5, 1.1, 1.2],
                 boundary_lambda=1.0,
                 orientation_weight=1.1):
        super().__init__()

        self.boundary_weights = boundary_weights
        self.orientation_weight = orientation_weight

        self.b_loss = CCELoss(b_lambda=boundary_lambda, reduction='mean')
        self.o_loss = OORLoss(reduction='mean')

    def forward(self, boundary_x_list, orientation_x, labels):
        boundary_y, orientation_y = labels[:, 0], labels[:, 1]

        b_blance_pos = float(torch.sum(boundary_y == 1)) / float(torch.numel(boundary_y))
        b_blance_neg = float(torch.sum(boundary_y == 0)) / float(torch.numel(boundary_y))
        b_blance = [b_blance_pos, b_blance_neg]

        assert len(boundary_x_list) == len(self.boundary_weights)

        boundary_losses = [b_w * self.b_loss(b_x, boundary_y, b_blance)
                           for b_x, b_w in zip(boundary_x_list, self.boundary_weights)]
        orientation_loss = self.orientation_weight * self.o_loss(orientation_x, orientation_y, boundary_y)

        return boundary_losses, orientation_loss
