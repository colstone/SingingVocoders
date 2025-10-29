import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Embedding
from typing import List
from itertools import repeat
from .chained_optimizer import ChainedOptimizer, OptimizerSpec

coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
]

# safety factor for numerical stability (but exclude last polynomial )
coeffs_list = [(a / 1.01 , b / 1.01**3 , c / 1.01**5) for (a, b, c) in coeffs_list[: -1]] + [coeffs_list[-1]]


def get_bf16_support_map():
    bf16_support_map = {}

    if not torch.cuda.is_available():
        return bf16_support_map

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return bf16_support_map

    for i in range(device_count):
        device = torch.device(f'cuda:{i}')       
        major, minor = torch.cuda.get_device_capability(device)
        bf16_support_map[device] = (major >= 8)
        
    return bf16_support_map
    
    
def zeropower_via_newtonschulz5(G: Tensor, steps: int, use_bf16: bool) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim == 3 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    #a, b, c = (3.4445, -4.7750,  2.0315)
    
    X = G.to(dtype = torch.bfloat16 if use_bf16 else torch.float32)

    # Ensure spectral norm is at most 1
    X = F.normalize(X, p=2.0, dim=(-2, -1), eps=1e-7)
    
    # Perform the NS iterations
    hs = coeffs_list[: steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
    if X.size(-2) < X.size(-1):
        for a, b, c in hs:
            A = torch.bmm(X, X.mT)
            A = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, A, X, beta=a, alpha=1)
    else:
        for a, b, c in hs:
            A = torch.bmm(X.mT, X)
            A = torch.baddbmm(A, A, A, beta=b, alpha=c)
            X = torch.baddbmm(X, X, A, beta=a, alpha=1)
            
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, params, lr=5e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.bf16_support_map = get_bf16_support_map()
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            shape_groups = {}
            for p in filter(lambda p: p.grad is not None, group["params"]):
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                key = (p.shape, p.device, p.dtype)
                if key not in shape_groups:
                    shape_groups[key] = {"params": [], "grads": [], "buffers": []}
                shape_groups[key]["params"].append(p)
                shape_groups[key]["grads"].append(g)
                shape_groups[key]["buffers"].append(state["momentum_buffer"])
            for key in shape_groups:
                group_data = shape_groups[key]
                p, g, buf, m = group_data["params"], group_data["grads"], group_data["buffers"], group["momentum"]
                torch._foreach_lerp_(buf, g, 1-m)
                if group["nesterov"]:
                    torch._foreach_lerp_(g, buf, m)
                    g = torch.stack(g)
                else:
                    g = torch.stack(buf)
                original_shape = g.shape
                if g.ndim >= 4:  # for the case of conv filters
                    g = g.view(g.size(0), g.size(1), -1)
                use_bf16 = self.bf16_support_map.get(g.device, False)
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"], use_bf16=use_bf16)
                if group["weight_decay"] > 0:
                    torch._foreach_mul_(p, 1 - group["lr"] * group["weight_decay"])
                torch._foreach_add_(p, g.view(original_shape).unbind(0), alpha=-group["lr"] * max(g[0].size()) ** 0.5)


def get_params_for_muon(model) -> List[Parameter]:
    """
    Filter parameters of a module into two groups: those that can be optimized by Muon,
    and those that should be optimized by a standard optimizer.
    Args:
        module: The module to filter parameters for.
    Returns:
        A list of parameters that should be optimized with muon.
    """
    muon_params = []
    for module in model.modules():
        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                continue
            if not isinstance(module, nn.Embedding) and param.ndim >= 2:
                muon_params.append(param)
    return muon_params


class Muon_AdamW(ChainedOptimizer):
    def __init__(self, model, lr=0.0005, weight_decay=0.0, muon_args={}, adamw_args={}, verbose=False):
        muon_params_id_set = set(id(p) for p in get_params_for_muon(model))
        spec_muon = OptimizerSpec(Muon, muon_args, lambda param: id(param) in muon_params_id_set)
        spec_adamw = OptimizerSpec(torch.optim.AdamW, adamw_args, None)
        specs = [spec_muon, spec_adamw]
        callback = None
        if verbose:
            callback = lambda p, spec_idx: print(
            f"Adding param {p.shape} to optimizer{spec_idx} {str(specs[spec_idx].class_type)}"
        )
        super().__init__(model.parameters(), specs, lr=lr, weight_decay=weight_decay, optimizer_selection_callback=callback)