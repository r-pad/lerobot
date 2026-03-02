# π₀.₅ (pi05)

This repository contains the Hugging Face port of **π₀.₅**, adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi) by the Physical Intelligence.
It is designed as a **Vision-Language-Action model with open-world generalization**.

---

## Model Overview

| Feature              | π₀                                                     | π₀.₅                                      |
| -------------------- | ------------------------------------------------------ | ----------------------------------------- |
| Time Conditioning    | Concatenates time with actions via `action_time_mlp_*` | Uses `time_mlp_*` for AdaRMS conditioning |
| AdaRMS               | Not used                                               | Used in action expert                     |
| Tokenizer Length     | 48 tokens                                              | 200 tokens                                |
| Discrete State Input | False (Uses `state_proj` layer)                        | True                                      |
| Parameter Count      | Higher (includes state embedding)                      | Lower (no state embedding)                |

---

## Vendored Transformers Changes

The upstream [LeRobot PI0.5 implementation](https://github.com/huggingface/lerobot) depends on a
[custom branch of transformers](https://github.com/huggingface/transformers/tree/fix/lerobot_openpi)
(`git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi`) that patches several
model files. To avoid that fork dependency, we vendor the necessary changes locally. The relevant
commit is [`3676fc8`](https://github.com/huggingface/transformers/commit/3676fc805228e63d9ae97b622d715af847c276c6).

### What was changed and why

1. **Gemma: AdaRMS support** (`gemma_config.py`, `gemma_modeling.py`)
   - Adds `use_adarms` and `adarms_cond_dim` to `GemmaConfig`.
   - `GemmaRMSNorm` gains an adaptive mode: when a conditioning signal (the diffusion timestep
     embedding) is provided, it produces scale/shift/gate via a learned dense layer instead of
     using a fixed learned weight. This is how PI0.5's action expert is conditioned on the
     flow-matching timestep.
   - `GemmaDecoderLayer` threads an `adarms_cond` tensor through to both layer norms.
   - `_gated_residual()` helper applies gate-modulated residual connections.
   - Only the action expert uses AdaRMS (`use_adarms=[False, True]`); the VLM backbone does not.

2. **PaliGemma: remove `hidden_size**0.5` scaling** (in `embed_image()`)
   - Stock PaliGemma divides projected image features by `sqrt(hidden_size)` (~55.4 for Gemma-2B).
     PI0.5 weights were trained without this normalization. Leaving it in makes image features
     ~55x too small, producing garbage outputs. We bypass `get_image_features()` and reimplement
     the projection without the scaling factor.

3. **Gemma: remove `hidden_states * sqrt(hidden_size)` normalizer** (monkey-patch in `modeling_pi05.py`)
   - Stock transformers scales token embeddings by `sqrt(hidden_size)` inside `GemmaModel.forward()`.
     The custom branch comments this out. PI0.5 weights were trained without it. We monkey-patch
     `GemmaModel.forward` with a copy that omits this line. This affects the VLM's internal Gemma
     (the action expert already uses our vendored Gemma which omits it).

4. **SigLIP: bfloat16 dtype cast** (monkey-patch in `modeling_pi05.py`)
   - PI0.5 runs in bfloat16, but SigLIP's patch/position embeddings are kept in float32 for
     precision. The stock SigLIP encoder does not cast these float32 hidden states before feeding
     them into the bfloat16 attention layers, causing dtype mismatches. The monkey-patch casts
     embedding outputs to bfloat16 when the encoder layers expect it.

### Compatibility note

These vendored files and patches have been tested with **transformers 4.56**. Later versions of
transformers may change the signatures or internals of `GemmaForCausalLM`,
`PaliGemmaForConditionalGeneration`, or `SiglipVisionTransformer`. If you upgrade transformers,
verify that:
- `GemmaModel.forward()` still applies the `hidden_states * normalizer` scaling (if removed
  upstream, the monkey-patch becomes a no-op but should be cleaned up)
- `GemmaDecoderLayer.forward()` still accepts the same kwargs
- `PaliGemmaForConditionalGeneration.model.vision_tower` and `.multi_modal_projector` still exist
- `SiglipVisionTransformer.embeddings.forward()` and `SiglipEncoder.layers[0].self_attn.q_proj`
  still exist and behave the same way

---

## Citation

If you use this work, please cite both **OpenPI** and the π₀.₅ paper:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{intelligence2025pi05visionlanguageactionmodelopenworld,
  title        = {π₀.₅: a Vision-Language-Action Model with Open-World Generalization},
  author       = {Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Manuel Y. Galliker and Dibya Ghosh and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and James Tanner and Quan Vuong and Homer Walke and Anna Walling and Haohuan Wang and Lili Yu and Ury Zhilinsky},
  year         = {2025},
  eprint       = {2504.16054},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.16054},
}
```

---

## License

This port follows the **Apache 2.0 License**, consistent with the original [OpenPI repository](https://github.com/Physical-Intelligence/openpi).
