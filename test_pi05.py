import torch
from lerobot.common.policies.pi05.configuration_pi05 import PI05Config
from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.types import FeatureType, PolicyFeature

# --- Config ---
config = PI05Config()
config.device = "cuda"
config.dtype = "bfloat16"  # halves VRAM (~7GB vs ~14GB for weights)
config.gradient_checkpointing = True  # trades compute for memory during training
config.input_features = {
    "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
}
config.output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
}

# --- Instantiate ---
policy = PI05Policy(config)
policy.eval()
device = next(policy.parameters()).device
print(f"Model: {sum(p.numel() for p in policy.parameters()):,} params on {device}")
vram_gb = torch.cuda.memory_allocated() / 1e9
print(f"VRAM after load: {vram_gb:.1f} GB")

# --- Dummy batch ---
B = 2
batch = {
    "observation.images.top": torch.randn(B, 3, 224, 224, device=device),
    "observation.state": torch.randn(B, 14, device=device),
    "observation.language.tokens": torch.randint(0, 1000, (B, 20), device=device),
    "observation.language.attention_mask": torch.ones(B, 20, dtype=torch.bool, device=device),
    "action": torch.randn(B, 50, 14, device=device),
}

# --- Training forward (loss) ---
print("\n--- Training forward ---")
policy.train()
loss, loss_dict = policy.forward(batch)
print(f"Training loss: {loss.item():.4f}")
print(f"Loss dict: {loss_dict}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

# --- Inference forward (predict actions) ---
print("\n--- Inference forward ---")
torch.cuda.reset_peak_memory_stats()
policy.eval()
with torch.no_grad():
    actions = policy.predict_action_chunk(batch)
print(f"Predicted actions shape: {actions.shape}")  # expect [B, 50, 14]
print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
