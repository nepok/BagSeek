from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn


DEFAULT_IMAGE_ROOT = Path("/mnt/data/bagseek/flask-backend/src/extracted_images_per_topic")
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/bagseek/flask-backend/src/embeddings_per_topic")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_RUNS: List[dict] = [
    {
        "name": "ViT-B-16-finetuned(09.10.25)",
        "checkpoint": "/mnt/data/bagseek/flask-backend/src/models/ViT-B-16-finetuned(09.10.25).pt",
        "image_root": DEFAULT_IMAGE_ROOT,
        "output_root": DEFAULT_OUTPUT_ROOT,
        "model_dir_name": "ViT-B-16-finetuned(09.10.25)",
        "batch_size": 64,
        "device": DEFAULT_DEVICE,
        "overwrite": False,
        "enabled": True,
    },
    {
        "name": "agriclip",
        "checkpoint": "/mnt/data/bagseek/flask-backend/src/models/CLIP-PT.pt",
        "image_root": DEFAULT_IMAGE_ROOT,
        "output_root": DEFAULT_OUTPUT_ROOT,
        "model_dir_name": "agriclip",
        "batch_size": 64,
        "device": DEFAULT_DEVICE,
        "overwrite": False,
        "enabled": True,
    },
]


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__(normalized_shape, eps=eps)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor | None = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def _apply_attn(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.attn_mask
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)
        result, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=mask, need_weights=False)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._apply_attn(x)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor | None = None):
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        grid = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(scale * torch.randn(grid + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # [batch, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # [batch, grid, width]

        class_embedding = self.class_embedding.to(x.dtype)
        batch_class = class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([batch_class, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x)
        x = self.ln_post(x[:, 0, :])  # CLS token

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vision_width: int,
        vision_layers: int,
        vision_patch_size: int,
        image_resolution: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()
        self.context_length = context_length

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
        )

        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads, attn_mask=self.build_text_mask(context_length))
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    @staticmethod
    def build_text_mask(context_length: int) -> torch.Tensor:
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self) -> torch.dtype:
        return self.positional_embedding.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.type(self.dtype))

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = self.transformer(x)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def count_layers(prefix: str, state_dict: dict[str, torch.Tensor]) -> int:
    layers = set()
    prefix_parts = prefix.split(".")
    index = len(prefix_parts)
    for key in state_dict.keys():
        if key.startswith(prefix):
            layer_id = key.split(".")[index]
            layers.add(int(layer_id))
    return len(layers)


def build_model_from_state_dict(state_dict: dict[str, torch.Tensor]) -> CLIP:
    dtype = state_dict["visual.class_embedding"].dtype
    if dtype == torch.float16 and not torch.cuda.is_available():
        state_dict = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
        dtype = torch.float32  # we will keep the model in float32 on CPU

    # Vision hyperparameters
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_patch_size = state_dict["visual.conv1.weight"].shape[2]
    num_pos_tokens = state_dict["visual.positional_embedding"].shape[0]
    grid_size = int((num_pos_tokens - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    vision_layers = count_layers("visual.transformer.resblocks", state_dict)

    # Text transformer hyperparameters
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = count_layers("transformer.resblocks", state_dict)

    model = CLIP(
        embed_dim=embed_dim,
        vision_width=vision_width,
        vision_layers=vision_layers,
        vision_patch_size=vision_patch_size,
        image_resolution=image_resolution,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
    )

    dtype = state_dict["visual.class_embedding"].dtype
    if dtype == torch.float16:
        model = model.half()
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _first_matching_key(container: dict, keys: Iterable[str]) -> dict | None:
    for key in keys:
        if key in container and isinstance(container[key], dict):
            return container[key]
    return None


def load_agriclip(checkpoint_path: str, device: str = "cpu") -> CLIP:
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict[str, torch.Tensor]

    if isinstance(raw, dict):
        candidate = _first_matching_key(raw, ("state_dict", "model_state_dict", "model", "ema_state_dict"))
        if candidate is not None:
            state_dict = candidate
        else:
            state_dict = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
            if not state_dict:
                raise ValueError(f"Unsupported checkpoint structure: keys={list(raw.keys())[:10]}")
    else:
        state_dict = raw  # assume direct state dict

    # Remove common wrappers/prefixes (e.g., 'model.', 'module.')
    for prefix in ("module.", "model.", "clip."):
        state_dict = _strip_prefix(state_dict, prefix)

    if "visual.class_embedding" not in state_dict:
        suspected_prefixes = {key.split(".", 1)[0] for key in state_dict.keys()}
        for prefix in suspected_prefixes:
            candidate = _strip_prefix(state_dict, f"{prefix}.")
            if "visual.class_embedding" in candidate:
                state_dict = candidate
                break

    if "visual.class_embedding" not in state_dict:
        raise KeyError(
            "Checkpoint does not contain CLIP visual weights under expected keys. "
            "Verify that the checkpoint was produced from a CLIP-compatible model."
        )

    model = build_model_from_state_dict(state_dict)
    model.to(torch.device(device))
    return model


def load_image_tensor(image_path: str, image_resolution: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_resolution, image_resolution), Image.BICUBIC)
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    image_np = (image_np - mean) / std
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(device=device, dtype=dtype)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(slots=True)
class EmbedTask:
    image_path: Path
    output_path: Path


def iter_image_files(image_root: Path) -> Iterable[Path]:
    for path in sorted(image_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        yield path


def build_tasks(image_root: Path, embedding_root: Path, overwrite: bool) -> List[EmbedTask]:
    tasks: List[EmbedTask] = []
    for image_path in iter_image_files(image_root):
        rel_path = image_path.relative_to(image_root)
        output_path = (embedding_root / rel_path).with_suffix(".pt")
        if not overwrite and output_path.exists():
            continue
        tasks.append(EmbedTask(image_path=image_path, output_path=output_path))
    return tasks


def batched(seq: Sequence[EmbedTask], batch_size: int) -> Iterable[Sequence[EmbedTask]]:
    for idx in range(0, len(seq), batch_size):
        yield seq[idx : idx + batch_size]


def ensure_parent_dirs(tasks: Sequence[EmbedTask]) -> None:
    for task in tasks:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)


def encode_batch(tasks: Sequence[EmbedTask], model, device: str) -> Tuple[torch.Tensor, List[EmbedTask]]:
    tensors: List[torch.Tensor] = []
    successful: List[EmbedTask] = []
    for task in tasks:
        try:
            tensor = load_image_tensor(
                image_path=str(task.image_path),
                image_resolution=model.visual.input_resolution,
                device=device,
                dtype=model.dtype,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to process {task.image_path}: {exc}")
            continue
        tensors.append(tensor)
        successful.append(task)

    if not tensors:
        output_dim = model.visual.proj.shape[1]
        return torch.empty(0, output_dim, device=device, dtype=model.dtype), []

    batch = torch.cat(tensors, dim=0)
    with torch.no_grad():
        features = model.encode_image(batch)
    return features, successful


def save_embeddings(batch_features: torch.Tensor, tasks: Sequence[EmbedTask], checkpoint_name: str) -> None:
    for idx, task in enumerate(tasks):
        embedding = batch_features[idx].detach().cpu()
        payload = {
            "embedding": embedding,
            "source_image": str(task.image_path),
            "model_checkpoint": checkpoint_name,
            "dtype": str(embedding.dtype),
            "shape": tuple(embedding.shape),
        }
        torch.save(payload, task.output_path)


def run_configuration(config: dict) -> None:
    checkpoint_value = config.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("Configuration missing required 'checkpoint' entry.")
    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_root_value = config.get("image_root", DEFAULT_IMAGE_ROOT)
    image_root = Path(image_root_value).expanduser()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist or is not a directory: {image_root}")

    output_root_value = config.get("output_root", DEFAULT_OUTPUT_ROOT)
    output_root = Path(output_root_value).expanduser()

    model_dir_name_value = config.get("model_dir_name")
    model_dir_name = model_dir_name_value or checkpoint_path.stem
    if not isinstance(model_dir_name, str):
        raise ValueError(f"model_dir_name must be a string or None, got {model_dir_name!r}")
    embedding_root = output_root / model_dir_name
    embedding_root.mkdir(parents=True, exist_ok=True)

    device = str(config.get("device") or DEFAULT_DEVICE)
    batch_size = int(config.get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    overwrite = bool(config.get("overwrite", False))
    job_name = str(config.get("name") or model_dir_name)

    print(f"[{job_name}] Loading model from {checkpoint_path} on device {device} ...")
    model = load_agriclip(str(checkpoint_path), device=device)

    tasks = build_tasks(image_root=image_root, embedding_root=embedding_root, overwrite=overwrite)
    total_images = sum(1 for _ in iter_image_files(image_root))
    remaining = len(tasks)

    print(f"[{job_name}] Found {total_images} image files under {image_root}.")
    if remaining == 0:
        print(f"[{job_name}] All embeddings already exist. Nothing to do.")
        return

    print(f"[{job_name}] Preparing to embed {remaining} images (skipping {total_images - remaining} already processed).")
    checkpoint_name = checkpoint_path.name

    completed = 0
    for batch_idx, batch_tasks in enumerate(batched(tasks, batch_size), start=1):
        ensure_parent_dirs(batch_tasks)
        features, valid_tasks = encode_batch(batch_tasks, model, device)
        if not valid_tasks:
            print(f"[{job_name}][{batch_idx}] No valid images in this batch; skipping.", flush=True)
            continue
        features = torch.nn.functional.normalize(features, dim=-1)
        save_embeddings(features, valid_tasks, checkpoint_name)
        completed += len(valid_tasks)
        print(f"[{job_name}][{batch_idx}] Saved embeddings for {completed}/{remaining} pending images.", flush=True)

    print(f"[{job_name}] Embedding complete.")


def main() -> int:
    for config in EMBEDDING_RUNS:
        if not config.get("enabled", True):
            name = config.get("name") or config.get("checkpoint")
            print(f"[{name}] Configuration disabled — skipping.")
            continue
        run_configuration(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
