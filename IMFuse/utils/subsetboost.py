import torch
import torch.nn.functional as F


MODALITY_NAMES = ("flair", "t1ce", "t1", "t2")

SUBSET_MASKS = (
    (True, False, False, False),
    (False, True, False, False),
    (False, False, True, False),
    (False, False, False, True),
    (True, True, False, False),
    (True, False, True, False),
    (True, False, False, True),
    (False, True, True, False),
    (False, True, False, True),
    (False, False, True, True),
    (True, True, True, False),
    (True, True, False, True),
    (True, False, True, True),
    (False, True, True, True),
    (True, True, True, True),
)


def _subset_name(mask):
    return "".join(name for name, present in zip(MODALITY_NAMES, mask) if present)


SUBSET_NAMES = tuple(_subset_name(mask) for mask in SUBSET_MASKS)


def mask_to_subset_indices(mask):
    mask = mask.bool()
    bits = torch.tensor([8, 4, 2, 1], device=mask.device, dtype=torch.long)
    codes = torch.matmul(mask.long(), bits)

    code_to_index = torch.full((16,), -1, device=mask.device, dtype=torch.long)
    subset_codes = []
    for subset in SUBSET_MASKS:
        subset_tensor = torch.tensor(subset, device=mask.device, dtype=torch.long)
        subset_codes.append(int(torch.matmul(subset_tensor, bits).item()))
    code_to_index[torch.tensor(subset_codes, device=mask.device)] = torch.arange(
        len(SUBSET_MASKS), device=mask.device
    )

    indices = code_to_index[codes]
    if torch.any(indices < 0):
        raise ValueError(f"Invalid empty modality subset mask: {mask}")
    return indices


def per_sample_seg_loss(output, target, num_cls, criterions):
    losses = []
    for sample_idx in range(output.size(0)):
        pred_i = output[sample_idx : sample_idx + 1]
        target_i = target[sample_idx : sample_idx + 1]
        cross_i = criterions.softmax_weighted_loss(pred_i, target_i, num_cls=num_cls)
        dice_i = criterions.dice_loss(pred_i, target_i, num_cls=num_cls)
        losses.append(cross_i + dice_i)
    return torch.stack(losses).view(-1)


def subset_cvar_loss(losses, fraction):
    if losses.numel() == 0:
        return losses.sum()
    k = max(1, int(torch.ceil(torch.tensor(losses.numel() * fraction)).item()))
    k = min(k, losses.numel())
    return torch.topk(losses, k=k, largest=True).values.mean()


def make_random_superset_mask(mask):
    mask = mask.bool()
    superset = mask.clone()
    valid = torch.zeros(mask.size(0), device=mask.device, dtype=torch.bool)
    for sample_idx in range(mask.size(0)):
        missing = torch.nonzero(~mask[sample_idx], as_tuple=False).view(-1)
        if missing.numel() == 0:
            continue
        choice = missing[torch.randint(missing.numel(), (1,), device=mask.device)]
        superset[sample_idx, choice] = True
        valid[sample_idx] = True
    return superset, valid


class SubsetRiskTracker:
    def __init__(self, momentum=0.95, device=None):
        self.momentum = momentum
        self.device = device or torch.device("cpu")
        self.loss_ema = torch.zeros(len(SUBSET_MASKS), device=self.device)
        self.counts = torch.zeros(len(SUBSET_MASKS), device=self.device)

    def state_dict(self):
        return {
            "momentum": self.momentum,
            "loss_ema": self.loss_ema.detach().cpu(),
            "counts": self.counts.detach().cpu(),
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.momentum = float(state.get("momentum", self.momentum))
        self.loss_ema = state["loss_ema"].to(self.device)
        self.counts = state["counts"].to(self.device)

    @torch.no_grad()
    def update(self, subset_indices, losses):
        subset_indices = subset_indices.detach().long()
        losses = losses.detach().float()
        for subset_idx, loss in zip(subset_indices.tolist(), losses.tolist()):
            if self.counts[subset_idx] == 0:
                self.loss_ema[subset_idx] = loss
            else:
                self.loss_ema[subset_idx] = (
                    self.momentum * self.loss_ema[subset_idx]
                    + (1.0 - self.momentum) * loss
                )
            self.counts[subset_idx] += 1

    def weak_indices(self, topk):
        observed = self.counts > 0
        if not torch.any(observed):
            return torch.empty(0, device=self.device, dtype=torch.long)
        scores = self.loss_ema.clone()
        scores[~observed] = -float("inf")
        k = min(int(topk), int(observed.sum().item()))
        return torch.topk(scores, k=k, largest=True).indices

    def weak_weighted_loss(self, losses, subset_indices, topk, weak_weight):
        weak = self.weak_indices(topk)
        if weak.numel() == 0:
            return losses.mean()
        is_weak = (subset_indices[:, None] == weak[None, :]).any(dim=1).float()
        weights = 1.0 + weak_weight * is_weak
        return torch.sum(losses * weights) / torch.clamp(weights.sum(), min=1.0)

    def summary(self, topk=5):
        weak = self.weak_indices(topk).detach().cpu().tolist()
        return ", ".join(
            f"{SUBSET_NAMES[idx]}:{self.loss_ema[idx].item():.4f}" for idx in weak
        )


def lattice_ranking_loss(subset_losses, superset_losses, valid, margin):
    if not torch.any(valid):
        return subset_losses.sum() * 0.0
    return F.relu(superset_losses[valid] - subset_losses.detach()[valid] + margin).mean()
