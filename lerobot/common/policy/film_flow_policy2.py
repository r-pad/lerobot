import torch
import torch.nn as nn
from monai.networks.nets import UNet
from collections import OrderedDict

# ======================================================
# Original Flow Policy
# ======================================================

class FlowPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet(
            spatial_dims=2,
            in_channels=2,
            out_channels=2,
            channels=(32, 64, 128, 256, 512),
            strides=(1, 1, 1, 1),
        )

    def forward(self, init_mask, cur_depth):
        x = torch.cat([init_mask, cur_depth], dim=1)
        return self.net(x)

# ======================================================
# FiLM
# ======================================================

class FiLM(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        return self.gamma * x + self.beta

# ======================================================
# Checkpoint key rewrite (if you need)
# ======================================================

def rewrite_monai_keys(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        nk = nk.replace(".sub0", ".submodule.0")
        nk = nk.replace(".sub1", ".submodule.1")
        nk = nk.replace(".sub2", ".submodule.2")
        nk = nk.replace(".subconv", ".submodule.conv")
        nk = nk.replace(".subadn", ".submodule.adn")
        new_sd[nk] = v
    return new_sd

# ======================================================
# Utilities: replace a submodule by name
# ======================================================

def _get_parent_module(root: nn.Module, full_name: str):
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def replace_module(root: nn.Module, full_name: str, new_module: nn.Module):
    parent, leaf = _get_parent_module(root, full_name)
    setattr(parent, leaf, new_module)

# ======================================================
# Locate bottleneck by one dry forward (temporary hooks)
# Strategy: among all Conv/ConvBlock-like modules that output C=512,
# pick the one with smallest H*W (deepest)
# ======================================================

@torch.no_grad()
def find_bottleneck_module_name(unet: nn.Module, device: torch.device,
                                in_shape=(1, 2, 240, 320), target_c=512):
    x = torch.randn(*in_shape, device=device)

    candidates = []  # (H*W, name)

    handles = []
    def make_hook(name):
        def hook(m, inp, out):
            # out could be Tensor or tuple/list
            if isinstance(out, (tuple, list)):
                return
            if not torch.is_tensor(out):
                return
            if out.dim() == 4 and out.shape[1] == target_c:
                hw = int(out.shape[2] * out.shape[3])
                candidates.append((hw, name))
        return hook

    # hook on modules likely to produce feature maps (Conv/ResidualUnit/etc.)
    for name, m in unet.named_modules():
        # Conv2d is safe to hook for locating (NOT for training)
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(make_hook(name)))

    _ = unet(x)

    for h in handles:
        h.remove()

    if not candidates:
        raise RuntimeError(f"Cannot find any module output with C={target_c}. "
                           f"Check UNet channels or input shape.")

    # smallest spatial size = deepest
    candidates.sort(key=lambda t: t[0])
    best_hw, best_name = candidates[0]
    return best_name

# ======================================================
# Inject FiLM after a target module (wrapper)
# ======================================================

class ModuleWithFiLM(nn.Module):
    def __init__(self, base: nn.Module, film: FiLM):
        super().__init__()
        self.base = base
        self.film = film

    def forward(self, x):
        y = self.base(x)
        # only apply if tensor 4D and channel matches
        if torch.is_tensor(y) and y.dim() == 4:
            y = self.film(y)
        return y

def inject_film_at_bottleneck(unet: nn.Module, device: torch.device,
                              in_shape=(1, 2, 240, 320), target_c=512):
    # 1) locate
    bn_name = find_bottleneck_module_name(unet, device=device, in_shape=in_shape, target_c=target_c)
    # 2) wrap that module
    parent, leaf = _get_parent_module(unet, bn_name)
    old = getattr(parent, leaf)

    film = FiLM(target_c).to(device)
    new = ModuleWithFiLM(old, film).to(device)

    replace_module(unet, bn_name, new)
    return bn_name, film

# ======================================================
# Build: load pretrained + inject bottleneck FiLM
# ======================================================

def build_film_flow_policy_bottleneck_only(ckpt_path, device,
                                           freeze_backbone=True,
                                           dummy_hw=(240, 320)):
    base_policy = FlowPolicy().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)

    base_policy.load_state_dict(sd, strict=True)
    print("✅ Pretrained FlowPolicy loaded")

    # Freeze backbone if desired
    if freeze_backbone:
        for p in base_policy.net.parameters():
            p.requires_grad = False

    # Inject FiLM at bottleneck (single insertion)
    B, H, W = 1, dummy_hw[0], dummy_hw[1]
    bn_name, film = inject_film_at_bottleneck(
        base_policy.net, device=device,
        in_shape=(B, 2, H, W), target_c=512
    )
    print(f"✅ Injected FiLM(512) after bottleneck module: {bn_name}")

    # Ensure FiLM is trainable
    for p in film.parameters():
        p.requires_grad = True

    return base_policy

# ======================================================
# Sanity test
# ======================================================

def test_forward_backward(model, device):
    model.train()
    B, H, W = 2, 240, 320
    init_mask = torch.randn(B, 1, H, W, device=device)
    cur_depth = torch.randn(B, 1, H, W, device=device)

    out = model(init_mask, cur_depth)
    loss = out.pow(2).mean()
    loss.backward()

    # print trainable params
    print("\nTrainable parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(" ", n, p.grad is not None)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_net_multitask_ckpt/multi_2_epoch_100.pt"

    model = build_film_flow_policy_bottleneck_only(
        ckpt_path, device,
        freeze_backbone=True,    # 先只训 FiLM，最稳
        dummy_hw=(240, 320)
    )

    # forward/backward sanity
    test_forward_backward(model, device)
