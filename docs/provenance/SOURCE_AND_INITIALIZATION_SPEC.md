# Source and initialization specification: `runs/lora_s0/weights.pt`

Machine-readable twin: [`lora_s0_provenance.json`](lora_s0_provenance.json) (embedded in every artifact bundle as `provenance`).
Nothing here approves the checkpoint. Everything marked **verified** was reproduced by running the script named beside it (2026-10-16).

## 1. The checkpoint
| | |
|---|---|
| Object | `gs://terrawatch-fire-ace-478304/runs/lora_s0/weights.pt` |
| SHA-256 | `7bd5498f04838fe19240da2b7339d794cc5fc3948074a0d826c33303d5c42dfa` (345,237,991 bytes) |
| GCS CRC32C | `dlSr1Q==`, identical for the live generation (21:38:20Z) and the soft-deleted original upload (19:51:38Z, five minutes after training ended). No MD5 was recorded. |
| Format | torch zip, `{"model": state_dict}`, 172 float32 tensors, 86,292,996 parameters; the pickle references only `OrderedDict`, `FloatStorage`, `_rebuild_tensor_v2` |
| Contains | ViT-B/16 vision tower, rank-8 LoRA (q, v) on zero-indexed blocks 8-11, a 2x512 head, a copy of the initial head (`head0_*`) |
| Does **not** contain | text tower, logit scale, authenticity head, calibration, OOD state, optimizer, configuration |

## 2. Architecture, LoRA, outputs, preprocessing (verified: `fire_assessor.checkpoint`, `scripts/reproduce_logits.py`)
- **Backbone** `ViT-B-16-quickgelu` (OpenCLIP). The OpenAI weights must be loaded with the QuickGELU configuration; plain `ViT-B-16` silently degrades them.
- **LoRA** rank 8, alpha 16 (class default, not stored in the checkpoint; confirmed because logits reproduce with it), scaling alpha/rank = 2, on the `q` and `v` projections of zero-indexed blocks 8-11 (blocks 9-12 in one-indexed text). Base weights are frozen and live under `attn.mha.*`.
- **Outputs** (verified against labels): logit 0 = `flames_visible`, logit 1 = `smoke_visible`. Raw, uncalibrated, multi-label logits; probability = sigmoid(logit / temperature) once a calibration exists. The 512-d embedding is L2-normalised; the 768-d block-8 CLS feature is pre-norm and used only by an authenticity head.
- **Preprocessing**: RGB; Resize 224 (bicubic, shortest side); CenterCrop 224; ToTensor; Normalize mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`.
- **Check**: `FireModel` built with `pretrained=None` loads the checkpoint with `strict=True`; CPU fp32 inference reproduces the stored GPU-export logits to within 0.0095 on 24 test images and 0.026 on all 1,310 calibration images.

## 3. Reconstructed training-source identity
- The run recorded `git_commit: "srchash:c535f0ae41041492"`. The convention is the first 16 hex characters of SHA-256 over the bytes of every `src/fire_assessor/*.py`, sorted by file name.
- Commit `8965c96` (the only commit; created after training) hashes to `srchash:8f5b2bd60decd042`. The declared hash is reproduced exactly by taking that commit, **removing `model_card.py`** (it did not exist yet) and **reversing two later edits in `train.py::export_features`** (CPU-to-GPU export fix). Run `python3 -I scripts/reproduce_source_hash.py <repo> 8965c96`. A 64-bit match is not a coincidence, so the training path (`train`, `lora.py`, `model.py`, `splits.py`) is byte-identical to that commit.
- **Limitations that remain**
  1. **No retrievable revision.** No commit, tag or archive of the training-time tree exists; the training VM was deleted. The tree is reproducible by construction, not recoverable as an artifact.
  2. The two reversed edits are known from the authoring session's history; only the hash match validates them.
  3. The hash covers `src/fire_assessor/*.py` only: not `scripts/` (including `build_dfire_manifest.py`, which generated the splits), not configuration, not dependency versions.
  4. The working tree has since changed (path-preserving exporter, artifact contract), so its source hash is no longer either value above. **Always reproduce against commit `8965c96`, never the working tree.**

## 4. Pretrained initialization
- Intended: open_clip `ViT-B-16-quickgelu` + tag `openai`, which resolves to Hugging Face `timm/vit_base_patch16_clip_224.openai`, revision `977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d`, file `open_clip_model.safetensors` (598,516,980 bytes, SHA-256 `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f` per Hugging Face LFS metadata; **not recomputed**, only byte ranges were fetched).
- Original OpenAI release: `.../5806e77c...df416f/ViT-B-16.pt` (sha256 is the path component). **Not compared.**
- **Verified** (`scripts/compare_pretrained.py`): 56 of 56 sampled frozen vision tensors (118 MB) are bit-identical to the reference. Compared: class/positional embedding, `proj`, `conv1`, `ln_pre`, `ln_post`, and every attention/MLP/LayerNorm tensor of **blocks 0, 5, 8 and 11**, including the base weights inside the LoRA blocks.
- **Unverified**: the other 96 of 152 visual tensors (blocks 1, 2, 3, 4, 6, 7, 9, 10), assumed unchanged because the training code froze every visual parameter; equality with OpenAI's original file; the base-weights file actually read at training time (no hash or cache manifest was recorded).
- Supporting evidence: the initial head rows align with the pretrained text embeddings (cosine 0.94 flames, 0.96 smoke), and the training log shows a Hugging Face Hub request and no "No pretrained weights loaded" warning.

## 5. Training data and configuration
- Manifest hash `c04b4b2f551957d759468cfa699cad827c394189bd65a65164d475b346dd2789` reproduced from the bucket manifest with `partition := partition_xsrc`. 21,527 D-Fire images, 10,670 near-duplicate clusters, none spanning partitions; train 9,169 (WEB 8,204, PublicDataset 965, no AoF), val/cal/test 1,310 each, held-out AoF source 8,428.
- Declared: lr 1e-4 (LoRA) / 1e-3 (head), weight decay 0.01, 10 epochs, batch 32, label smoothing 0.05, patience 3, seed 0. Checkpoint selected on validation macro PR-AUC only (best epoch 9, 0.98727). Derived from the hash-reproduced source: AdamW, 5% warm-up + cosine, AMP on CUDA, BCE-with-logits, and the augmentation listed in the provenance record.

## 6. Missing training-environment information
**The training environment is not recorded anywhere and the run must not be described as reproducible.** Missing: Python, torch/torchvision/CUDA/driver versions, `open_clip_torch` version during training (the audit used 3.3.0), a `pip freeze`/lockfile, determinism settings, an artifact identifying the hardware (an NVIDIA L4 appears only in session history), complete stdout (stored logs are progress-bar output), and the base-weights hash read at training time. Seed 0 has not been re-run, so neither bit-exact nor statistical reproducibility is established.
