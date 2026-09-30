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
# FiLM module
# ======================================================

class FiLM(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        return self.gamma * x + self.beta


# ======================================================
# FiLM-adapted Flow Policy
# ======================================================

class FiLMFlowPolicy(nn.Module):
    def __init__(self, pretrained_unet):
        super().__init__()

        # Backbone
        self.net = pretrained_unet
        for p in self.net.parameters():
            p.requires_grad = False

        # FiLM layers (per channel width)
        self.films = nn.ModuleList([
            FiLM(32),
            FiLM(64),
            FiLM(128),
            FiLM(256),
            FiLM(512),
        ])

        self._register_film_hooks()

    def _register_film_hooks(self):
        self.handles = []

        for module in self.net.modules():
            if isinstance(module, nn.Conv2d):
                out_ch = module.out_channels
                for film in self.films:
                    if film.gamma.shape[1] == out_ch:
                        h = module.register_forward_hook(
                            self._make_film_hook(film)
                        )
                        self.handles.append(h)

    def _make_film_hook(self, film):
        def hook(module, inp, out):
            return film(out)
        return hook

    def forward(self, init_mask, cur_depth):
        x = torch.cat([init_mask, cur_depth], dim=1)
        return self.net(x)


# ======================================================
# Checkpoint key rewrite (MONAI compatibility)
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
# Load pretrained model + build FiLM model
# ======================================================

def build_film_flow_policy(ckpt_path, device):
    # -------- load original model --------
    base_policy = FlowPolicy().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"]
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)

    base_policy.load_state_dict(sd, strict=True)
    base_policy.eval()

    print("✅ Pretrained FlowPolicy loaded")

    # -------- wrap with FiLM --------
    film_policy = FiLMFlowPolicy(base_policy.net).to(device)
    film_policy.eval()

    print("✅ FiLMFlowPolicy initialized")

    return film_policy


# ======================================================
# Sanity test
# ======================================================

def test_film_policy(model, device):
    B, H, W = 2, 240, 320
    init_mask = torch.randn(B, 1, H, W, device=device)
    cur_depth = torch.randn(B, 1, H, W, device=device)

    with torch.no_grad():
        flow = model(init_mask, cur_depth)

    print("Output shape:", flow.shape)
    assert flow.shape == (B, 2, H, W)

    print("\nTrainable parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(" ", n)

    print("\nFiLM parameter shapes:")
    for i, film in enumerate(model.films):
        print(
            f"FiLM[{i}] "
            f"gamma {tuple(film.gamma.shape)} "
            f"beta {tuple(film.beta.shape)}"
        )

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nTotal trainable scalar parameters:", total)


# ======================================================
# Example usage
# ======================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_net_multitask_ckpt/multi_2_epoch_100.pt"  # ← change this

    film_policy = build_film_flow_policy(ckpt_path, device)
    test_film_policy(film_policy, device)
