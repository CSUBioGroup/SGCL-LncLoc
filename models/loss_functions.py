import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.2, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        input:
            features: sample features, whose size is [batch_size, hidden_dim].
            labels: ground truth labels for each sample, whose size is [batch_size].
            mask: mask used for contrastive learning, and the size is [batch_size, batch_size]. If sample i and j belong
                  to the same label, then mask_{i,j} = 1.
        output:
            the value of loss
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        features = F.normalize(features, p=2, dim=1)

        batch_size = features.shape[0]

        # About labels

        # Labels and mask cannot be defined at the same time.
        # Because if there is labels, then the mask must be obtained from the labels.
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        # If there is no labels and no mask, it is unsupervised learning.
        # Mask is a matrix with a diagonal of 1, indicating that (i, i) belongs to the same class.
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        # If labels are given, then the mask is given from the labels.
        # Mask_{i,j} = 1 when the labels of two samples i and j are the same.
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        For example: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # When labels of two samples i and j are the same, mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the dot product similarity between pairwise samples.
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)

        '''
        logits is the final similarity obtained by subtracting the maximum value of each row from anchor_dot_contrast.
        For example: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''

        # Get mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        '''
        For the calculation of Loss, 
        the position (i,i) represents the similarity of the sample itself,
        which is useless for Loss, 
        so it must mask itself.
        # Fill (i, i) with 0.
        logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''

        num_positives_per_row = torch.sum(positives_mask,
                                          axis=1)  # Number of positive samples besides itself [2 0 2 2].
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        '''
        Calculate the average log-likelihood of the positive samples.
        Given that a category may have only one sample, there is no positive sample.
        For example, our second category of labels is labels[1,2,1,1].
        So we only calculate loss if the number of positive samples > 0
        '''
        if num_positives_per_row.sum() == 0:
            return torch.tensor(0.)
        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
            num_positives_per_row > 0]

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
