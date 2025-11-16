import torch
import torch.nn as nn
import torch.nn.functional as F

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size

def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self):
        super(ContrastiveCorrelationLoss, self).__init__()

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2):
        # with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
        fd = tensor_correlation(norm(f1), norm(f2))

        # cd = tensor_correlation(norm(c1), norm(c2))
        loss = fd

        return loss

    def forward(self,
                fea_h: torch.Tensor,
                fea_v: torch.Tensor,
                # code_h: torch.Tensor,
                # code_v: torch.Tensor,
                ):

        coord_shape = [fea_v.shape[0], 64, 64, 2]


        coords1 = torch.rand(coord_shape, device=fea_h.device) * 2 - 1
        # coords2 = torch.rand(coord_shape, device=fea_v.device) * 2 - 1

        fea_h = sample(fea_h, coords1)
        fea_v = sample(fea_v, coords1)
        # code_h = sample(code_h, coords1)
        # code_v = sample(code_v, coords1)

        pos_intra_loss = self.helper(
            fea_h, fea_v)

        # neg_losses = []
        # neg_cds = []
        # for i in range(5):
        #     perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
        #     feats_neg = sample(orig_feats[perm_neg], coords2)
        #     code_neg = sample(orig_code[perm_neg], coords2)
        #     neg_inter_loss, neg_inter_cd = self.helper(
        #         feats, feats_neg, code, code_neg)
        #     neg_losses.append(neg_inter_loss)
        #     neg_cds.append(neg_inter_cd)
        # neg_inter_loss = torch.cat(neg_losses, axis=0)
        # neg_inter_cd = torch.cat(neg_cds, axis=0)

        return pos_intra_loss.mean() #, neg_inter_loss