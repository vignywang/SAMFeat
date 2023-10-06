import torch
import torch.nn as nn
import torch.nn.functional as f

class SAMFeat(nn.Module):
    def __init__(self, desc_dim):
        super(SAMFeat, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Shared Encoder.
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.dist_conv_feat = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dist_conv_fushion = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.heatmap1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.heatmap4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1.data.fill_(0.1)
        self.fuse_weight_2.data.fill_(0.2)
        self.fuse_weight_3.data.fill_(0.3)
        self.fuse_weight_4.data.fill_(0.4)

        self.scalemap = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.active = f.softplus
        self.conv_des = nn.Conv2d(384, desc_dim, kernel_size=3, stride=1, padding=1)
        self.conv_det = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.conv_edg = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)

        self.egde_module = EdgeAttentionModule(384, 10000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        c1 = self.relu(self.conv1b(x))  # 64 400×400

        c2 = self.pool(c1)
        c2 = self.relu(self.conv2a(c2))
        c2 = self.relu(self.conv2b(c2))  # 64 200×200

        c3 = self.pool(c2)
        c3 = self.relu(self.conv3a(c3))
        c3 = self.relu(self.conv3b(c3))  # 128 100×100

        c4 = self.pool(c3)
        c4 = self.relu(self.conv4a(c4))
        c4 = self.relu(self.conv4b(c4))  # 128 50×50

        dist_feat= self.dist_conv_feat(c4) # 256 50×50
        dist_feat_fushion = self.dist_conv_fushion(dist_feat) # 128 50×50
        c4 = c4 + dist_feat_fushion

        # Descriptor
        des_size = c3.shape[2:]  # 1/4 HxW
        d1 = f.interpolate(c1, des_size, mode='bilinear')
        d2 = f.interpolate(c2, des_size, mode='bilinear')
        d3 = c3
        d4 = f.interpolate(c4, des_size, mode='bilinear')
        feature = torch.cat((d1, d2, d3, d4), dim=1)

        # edge attention map
        edgemap = f.sigmoid(self.conv_edg(feature))

        # KeyPoint Map
        c3 = c3 * edgemap + c3
        heatmap1 = self.heatmap1(c1)
        heatmap2 = self.heatmap2(c2)
        heatmap3 = self.heatmap3(c3)
        heatmap4 = self.heatmap4(c4)
        des_size = heatmap1.shape[2:]  # 1/4 HxW
        heatmap2 = f.interpolate(heatmap2, des_size, mode='bilinear')
        heatmap3 = f.interpolate(heatmap3, des_size, mode='bilinear')
        heatmap4 = f.interpolate(heatmap4, des_size, mode='bilinear')
        heatmap = heatmap1 * self.fuse_weight_1 + heatmap2 * self.fuse_weight_2 + heatmap3 * self.fuse_weight_3 + heatmap4 * self.fuse_weight_4

        edge_feature = feature*edgemap
        feature =  self.egde_module(edge_feature, feature)

        # attention map
        meanmap = torch.mean(feature, dim=1, keepdim=True)
        attmap = self.scalemap(meanmap)
        attmap = self.active(attmap)

        # descriptors
        descriptor = feature
        descriptor = self.conv_des(descriptor)

        return heatmap, descriptor, attmap, dist_feat, edgemap


class EdgeAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (out_channels ** 0.5)

    def forward(self, edge_feature, feature_map):
        query = self.query_conv(edge_feature)
        key = self.key_conv(edge_feature)
        value = self.value_conv(edge_feature)

        # Compute attention scores using dot product between query_edge and key_feature
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale

        # Apply softmax to obtain attention weights
        attention_weights = f.softmax(attention_scores, dim=-1)

        # Compute the weighted sum of value_feature vectors using attention weights
        attended_values = torch.matmul(attention_weights, value)
        # Output_feature_map now contains the fused feature map
        output_feature_map = (feature_map + attended_values)

        return output_feature_map