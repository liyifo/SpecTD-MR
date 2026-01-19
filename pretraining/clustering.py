import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.cluster import KMeans


def weighted_cluster_average(node_Q, node_feat, edge_Q, edge_feat):
    node_weight_sum = torch.sum(node_Q, dim=0)
    node_cluster_feat = torch.mm(node_Q.T, node_feat)
    node_cluster_feat /= node_weight_sum.unsqueeze(1).clamp_min(1e-6)

    edge_weight_sum = torch.sum(edge_Q, dim=0)
    edge_cluster_feat = torch.mm(edge_Q.T, edge_feat)
    edge_cluster_feat /= edge_weight_sum.unsqueeze(1).clamp_min(1e-6)

    return node_cluster_feat, edge_cluster_feat


class DEC(nn.Module):
    def __init__(self, num_cluster=4, feat_dim=48):
        super(DEC, self).__init__()
        self.feat_dim = feat_dim
        self.num_cluster = num_cluster
        self.mean = nn.Parameter(torch.Tensor(num_cluster, self.feat_dim))
        init.kaiming_normal_(self.mean, mode='fan_in', nonlinearity='relu')
        self.Q = None

    def build_Q(self, node_feat):
        epsilon = 1.0
        Z = node_feat.unsqueeze(1)  # Shape: [N, 1, dim]
        diff = Z - self.mean  # Broadcasting subtraction
        squared_norm = torch.sum(diff ** 2, dim=2)  # Shape: [N, K]
        Q = torch.pow(squared_norm / epsilon + 1.0, -(epsilon + 1.0) / 2.0)
        return Q / torch.sum(Q, dim=1, keepdim=True)

    def loss(self, node_feat, epoch):
        if epoch == 0:
            self.init_mean(node_feat)
        self.Q = self.build_Q(node_feat)
        P = self.get_P()
        loss_c = torch.mean(P * torch.log(P / self.Q.clamp_min(1e-6)))
        return loss_c

    def init_mean(self, node_feat):
        kmeans = KMeans(n_clusters=self.num_cluster, n_init=10).fit(node_feat.cpu().detach().numpy())
        cluster_centers_tensor = torch.tensor(kmeans.cluster_centers_).to(node_feat.device)
        with torch.no_grad():
            self.mean.copy_(cluster_centers_tensor)

    def get_P(self):
        f_k = torch.sum(self.Q, dim=0)
        numerator = self.Q**2 / f_k
        denominator_terms = self.Q ** 2 / f_k.unsqueeze(0)
        denominator = torch.sum(denominator_terms, dim=1, keepdim=True)
        return numerator / denominator

    def predict(self):
        indices = torch.argmax(self.Q, dim=1)
        one_hot = F.one_hot(indices, num_classes=self.Q.shape[1])
        return one_hot

    def get_Q(self):
        return self.Q


class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PredictionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        return self.fc(x)

    def triplet_loss(self, anchor, positive, negative, margin=2):
        cos_sim_pos = self.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
        cos_sim_neg = self.cosine_similarity(anchor.unsqueeze(0), negative.unsqueeze(0))

        distance_positive = 1 - cos_sim_pos
        distance_negative = 1 - cos_sim_neg

        losses = torch.relu(distance_positive - distance_negative + margin)
        return losses.mean()

    def build_loss(self, node_cluster_feat, edge_cluster_feat):
        if node_cluster_feat.shape[0] < 2:
            return torch.tensor(0.0, device=node_cluster_feat.device, requires_grad=True)
        embedded_X_s = self(node_cluster_feat)
        embedded_X_t = self(edge_cluster_feat)

        total_loss = 0
        for i in range(embedded_X_s.shape[0]):
            distances = [-self.cosine_similarity(embedded_X_s[i].unsqueeze(0), t.unsqueeze(0)) for t in embedded_X_t]
            positive_index = distances.index(min(distances))
            for j in range(embedded_X_t.shape[0]):
                if j != positive_index:
                    total_loss += self.triplet_loss(embedded_X_s[i], embedded_X_t[positive_index], embedded_X_t[j])

        if embedded_X_s.shape[0] > 0:
            total_loss /= embedded_X_s.shape[0]
        return total_loss
