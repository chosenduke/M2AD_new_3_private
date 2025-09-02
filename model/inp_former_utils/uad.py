import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class INP_Former(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
            learnable_fuse=False,
            gather_weight=1.0,
            diversity_weight=0.0,
    ) -> None:
        super(INP_Former, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]
        self.learnable_fuse = learnable_fuse
        self.gather_weight = gather_weight
        self.diversity_weight = diversity_weight

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        # learnable fusion weights
        if self.learnable_fuse:
            # for fusing all target layer features into x
            self.enlist_fuse_w = nn.Parameter(torch.ones(len(self.target_layers)))
            # for encoder/decoder group fuse at output sides
            self.encoder_fuse_w = nn.ParameterList([
                nn.Parameter(torch.ones(len(group))) for group in self.fuse_layer_encoder
            ])
            self.decoder_fuse_w = nn.ParameterList([
                nn.Parameter(torch.ones(len(group))) for group in self.fuse_layer_decoder
            ])


    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    def diversity_loss(self, prototypes):
        # prototypes: (B, T, C)
        if prototypes.dim() != 3:
            return prototypes.new_zeros(())
        B, T, C = prototypes.shape
        if T <= 1:
            return prototypes.new_zeros(())
        p = F.normalize(prototypes, dim=-1)
        # similarity matrix for each batch
        sim = torch.bmm(p, p.transpose(1, 2))  # (B, T, T)
        # remove diagonal
        eye = torch.eye(T, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        off_diag = sim * (1.0 - eye)
        # penalize positive similarities (encourage orthogonality)
        loss = F.relu(off_diag).mean()
        return loss

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        # fuse features across all target layers
        if self.learnable_fuse:
            x = self.fuse_feature(en_list, weights=self.enlist_fuse_w)
        else:
            x = self.fuse_feature(en_list)

        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            if agg_prototype.dim() == 2:  # (T, C) -> (B, T, C) for the first aggregation
                agg_input = agg_prototype.unsqueeze(0).repeat(B, 1, 1)
            else:  # already (B, T, C)
                agg_input = agg_prototype
            agg_prototype = blk(agg_input, x)
        # for i, blk in enumerate(self.aggregation):
        #     if agg_prototype.dim() == 2:              # (T, C) 第一次聚合
        #         agg_input = agg_prototype.unsqueeze(0).repeat(B, 1, 1)  # → (B, T, C)
        #     else:                                     # (B, T, C) 第二次及以后
        #         agg_input = agg_prototype
        #     agg_prototype = blk(agg_input, x)

        # losses: patch-to-prototype gather and prototype diversity
        g_loss = self.gather_loss(x, agg_prototype) * self.gather_weight
        d_loss = self.diversity_loss(agg_prototype) * self.diversity_weight

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        if self.learnable_fuse:
            en = [self.fuse_feature([en_list[idx] for idx in idxs], weights=self.encoder_fuse_w[i])
                  for i, idxs in enumerate(self.fuse_layer_encoder)]
            de = [self.fuse_feature([de_list[idx] for idx in idxs], weights=self.decoder_fuse_w[i])
                  for i, idxs in enumerate(self.fuse_layer_decoder)]
        else:
            en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
            de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        total_aux_loss = g_loss + d_loss
        return en, de, total_aux_loss

    def fuse_feature(self, feat_list, weights=None):
        x = torch.stack(feat_list, dim=1)  # (B, K, N, C)
        if weights is None:
            return x.mean(dim=1)
        w = torch.softmax(weights, dim=0).view(1, -1, 1, 1)
        return (x * w).sum(dim=1)

    # def fuse_feature(self, feat_list):
    #     x = torch.stack(feat_list, dim=1)                  # (B, K, N, C)
    #     K = x.size(1)
    #     w = torch.linspace(1.0, 2.0, steps=K, device=x.device)  # 后层更大
    #     w = w / w.sum()
    #     return (x * w.view(1, K, 1, 1)).sum(dim=1)







































