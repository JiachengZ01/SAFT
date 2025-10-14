import torch
from models.model import multiGPU_CLIP_classwise, clip_img_preprocessing, multiGPU_CLIP

lower_limit, upper_limit = 0, 1


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def pgd_CLIP(
    model,
    X,
    target,
    text_tokens,
    alpha,
    attack_iters,
    norm,
    device,
    ncaps,
    epsilon=2 / 255,
):
    """
    PGD attack for CLIP model.
    """
    delta = torch.zeros_like(X).to(device)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        r = torch.zeros_like(delta.view(delta.size(0), -1).norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)).uniform_(0, 1)
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    target = target.repeat(ncaps)

    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, device)

        logits_per_image, _, _ = multiGPU_CLIP_classwise(model, _images, text_tokens)
        _, _, n_class = logits_per_image.shape
        logits_per_image = logits_per_image.view(-1, n_class)  # (ncaps * bs, n_class)
        CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction="mean").to(device)
        loss = CrossEntropyLoss(logits_per_image, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (
                (d + scaled_g * alpha)
                .view(d.size(0), -1)
                .renorm(p=2, dim=0, maxnorm=epsilon)
                .view_as(d)
            )
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()
    return delta
