import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastLoss(nn.Module):
    def __init__(self, batch_size, dim_in):
        super(ContrastLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = 1

        self.criterion = nn.CrossEntropyLoss().float()
        self.head = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_in, kernel_size=1, stride=1)
        )

    def Info_nce_loss(self, feature_s, feature_t):
        # feature_s = F.normalize(feature_s, dim=1)
        # feature_t = F.normalize(feature_t, dim=1)

        similarity_matrix = torch.matmul(feature_s.float(), feature_s.float().T)
        feature_t = torch.matmul(feature_t.float(), feature_t.float().T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(feature_t.shape[0], dtype=torch.bool)
        feature_t = feature_t[~mask].view(feature_t.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[feature_t.bool()].view(-1, 1)  # N * 1
        negatives = similarity_matrix[~feature_t.bool()] # N * k
        # print("positives.shape[0] ", positives.shape[0])
        bia = 0
        if positives.shape[0] != 0:
            bia = negatives.shape[0] % positives.shape[0]
            negatives = negatives[:(negatives.shape[0] - bia)].view(positives.shape[0], -1)
        else:
            positives = similarity_matrix[feature_t.bool()].view(2, 0)
            negatives = negatives[:(negatives.shape[0] - bia)].view(2, -1)

        logits = torch.cat([positives, negatives], dim=1) # N * (1 + k)
        feature_t = torch.zeros(positives.shape[0], dtype=torch.long)
        logits = logits / self.temperature
        loss = self.criterion(logits.cuda(), feature_t.cuda())

        return loss

    def forward(self, feat_s, feat_t):
        feat_s = F.normalize(self.head(feat_s), dim=1)
        feat_s = feat_s.view(feat_s.shape[0], -1)
        feat_t = F.normalize(self.head(feat_t), dim=1)
        feat_t = feat_t.view(feat_t.shape[0], -1)
        loss = self.Info_nce_loss(feat_s, feat_t)
        # feat = F.normalize(self.head(feat_s), dim=1).view(feat_s.shape[0], -1)
        # loss = self.Info_nce_loss(feat_s, feat_t)

        return loss



if __name__ == "__main__":
    """
    Sample usage. In order to test the code, Input and GT are randomly populated with values.
    """

    # Parameters for creating random input


    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
