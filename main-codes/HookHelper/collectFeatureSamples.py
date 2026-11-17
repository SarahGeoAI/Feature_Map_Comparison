import torch
import pandas as pd
import hashlib
from typing import Optional, Sequence, Dict, Any, Tuple, List
import random
import os

# ---------------------------
# Hook helper
# ---------------------------
class FeatureHook:
    """
    Attach a forward hook to any module to capture its output.
    Use as a context manager or manually call .close().
    """
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.handle = module.register_forward_hook(self._hook)
        self.features = None  # set on forward

    def _hook(self, module, inp, out):
        self.features = out

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------
# Layer resolver
# ---------------------------
def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Resolve a child module by dotted path, e.g., "decoder.final_block" or "layers.3".
    """
    cur = model
    for part in path.split("."):
        if part.isdigit():  # support Sequential/list modules
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    if not isinstance(cur, torch.nn.Module):
        raise TypeError(f"Resolved object at '{path}' is not an nn.Module")
    return cur


# ---------------------------
# Deterministic per-image RNG
# ---------------------------
def _stable_hash_to_int(s: str) -> int:
    """64-bit integer from SHA1 (stable across processes/python versions)."""
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "big", signed=False)

def _seeded_generator(base_seed: int, image_key: str) -> torch.Generator:
    g = torch.Generator(device="cpu")
    mixed = (base_seed & ((1 << 63) - 1)) ^ _stable_hash_to_int(image_key)
    g.manual_seed(mixed % (1 << 63))
    return g


# ---------------------------
# Main function
# ---------------------------
@torch.no_grad()
def collect_feature_samples(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    feature_layer: torch.nn.Module = None,
    feature_layer_path: Optional[str] = None,
    samples_per_image: int = 64,
    fm_prefix: str = "fm",
    csv_out: str = "feature_samples.csv",
    device: Optional[torch.device] = None,
    include_pred: bool = True,
    ignore_index: Optional[int] = None,
    rng_seed: Optional[int] = 1234,    # global seed (non-deterministic bits not tied to image_key)
    add_coords: bool = True,
    base_seed: int = 12345,            # master seed used to derive per-image deterministic RNGs
    id_from: str = "global_index",     # "global_index" or "meta:<key>" (e.g., "meta:path")
) -> str:
    """
    Run a trained model and deterministically sample per-pixel feature vectors from a hooked layer.

    Assumes dataloader yields (images, labels) or (images, labels, meta_dict).
    If using id_from="meta:<key>", meta_dict must contain that key (e.g., "path").

    Columns in output CSV:
      batch_idx, image_idx, [image_key], [y, x], label, [pred, correct], {fm_prefix_000 .. fm_prefix_C-1}
    """
    # --- argument checks
    assert (feature_layer is None) ^ (feature_layer_path is None), \
        "Provide exactly one of feature_layer or feature_layer_path"

    if rng_seed is not None:
        random.seed(rng_seed)
        torch.manual_seed(rng_seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Resolve the layer to hook
    layer = get_module_by_path(model, feature_layer_path) if feature_layer_path is not None else feature_layer

    rows: List[Dict[str, Any]] = []
    global_image_counter = 0
    meta_key = id_from.split(":", 1)[1] if id_from.startswith("meta:") else None

    with FeatureHook(layer) as fh:
        for batch_idx, batch in enumerate(dataloader):
            # Support (images, labels) or (images, labels, meta)
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Dataloader must return (images, labels) or (images, labels, meta_dict).")

            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            meta = batch[2] if (len(batch) >= 3 and isinstance(batch[2], dict)) else None

            # Forward pass -> fills fh.features
            logits = model(images)  # expected [N, K, H, W] for segmentation
            if fh.features is None:
                raise RuntimeError("Hook did not capture any features. Check the chosen layer.")

            feats = fh.features  # [N, C, Hf, Wf]
            # --- Fix for models that output [B, H, W, C] instead of [B, C, H, W]
            if feats.ndim == 4 and feats.shape[1] == feats.shape[2] and feats.shape[-1] < feats.shape[1]:
                # likely [B, H, W, C]
                feats = feats.permute(0, 3, 1, 2).contiguous()  # → [B, C, H, W]
            elif feats.ndim == 3:
                # Case 2: [B, L, C] → reshape to [B, C, H, W]
                B, L, C = feats.shape
                H = W = int(L ** 0.5)
                if H * W != L:
                    raise ValueError(f"Cannot reshape flattened features (L={L}) into square map.")
                feats = feats.permute(0, 2, 1).contiguous().view(B, C, H, W)    
            elif feats.ndim == 4:
                pass  # already [B, C, H, W]            
            else:
                raise ValueError(f"Unexpected feature shape: {feats.shape}")

            # If feature map size differs from labels, resize features to label shape
            if feats.shape[-2:] != labels.shape[-2:]:
                feats = torch.nn.functional.interpolate(
                    feats, size=labels.shape[-2:], mode="nearest"
                )

            N, C, H, W = feats.shape
            assert labels.shape[-2:] == (H, W), "Label and feature spatial dims must match."

            # Optionally compute predictions for reference
            if include_pred:
                if logits.shape[-2:] != (H, W):
                    logits = torch.nn.functional.interpolate(
                        logits, size=(H, W), mode="bilinear", align_corners=False
                    )
                preds = torch.argmax(logits, dim=1)  # [N, H, W]

            # Move to CPU for cheap indexed reads and to free VRAM
            feats_cpu = feats.detach().to("cpu")                 # [N, C, H, W]
            labels_cpu = labels.detach().to("cpu")               # [N, H, W]
            preds_cpu = preds.detach().to("cpu") if include_pred else None

            # Per-image deterministic sampling
            for n in range(N):
                # Derive stable image_key
                if meta_key is not None:
                    if meta is None or meta_key not in meta:
                        raise ValueError(f"id_from='{id_from}' but meta key '{meta_key}' is missing from batch[2].")
                    mk_val = meta[meta_key]
                    # support list/tuple/tensor/scalar
                    if isinstance(mk_val, (list, tuple)):
                        image_key_val = mk_val[n]
                    elif torch.is_tensor(mk_val):
                        image_key_val = mk_val[n].item() if mk_val.ndim > 0 else mk_val.item()
                    else:
                        image_key_val = mk_val
                    image_key = str(image_key_val)
                else:
                    image_key = f"global_index:{global_image_counter}"

                gen = _seeded_generator(base_seed, image_key)

                if ignore_index is None:
                    total_pixels = H * W
                    k = min(samples_per_image, total_pixels)
                    # Use the per-image generator for deterministic selection
                    idxs = torch.randperm(total_pixels, generator=gen)[:k]
                    ys = (idxs // W).tolist()
                    xs = (idxs % W).tolist()
                else:
                    mask = (labels_cpu[n] != ignore_index)
                    valid_yx = torch.nonzero(mask, as_tuple=False)  # [M, 2] (y, x)
                    if valid_yx.numel() == 0:
                        global_image_counter += 1
                        continue
                    k = min(samples_per_image, valid_yx.shape[0])
                    perm = torch.randperm(valid_yx.shape[0], generator=gen)[:k]
                    sample_yx = valid_yx[perm]
                    ys = sample_yx[:, 0].tolist()
                    xs = sample_yx[:, 1].tolist()

                # Extract feature vectors at selected pixels
                # feats_cpu[n]: [C,H,W] -> [H,W,C] then index
                f_hw_c = feats_cpu[n].permute(1, 2, 0)  # [H, W, C]
                sel = f_hw_c[ys, xs]                    # [k, C]
                labs = labels_cpu[n, ys, xs]            # [k]
                preds_k = preds_cpu[n, ys, xs] if include_pred else None

                # Build rows
                for i in range(len(ys)):
                    row: Dict[str, Any] = {
                        "batch_idx": batch_idx,
                        "image_idx": n,
                        "image_key": image_key,
                        "label": int(labs[i]),
                    }
                    if include_pred:
                        row["pred"] = int(preds_k[i])
                        row["correct"] = int(preds_k[i] == labs[i])
                    if add_coords:
                        row["y"] = int(ys[i])
                        row["x"] = int(xs[i])

                    fv = sel[i].tolist()
                    for c in range(C):
                        row[f"{fm_prefix}_{c:03d}"] = float(fv[c])

                    rows.append(row)

                global_image_counter += 1  # advance after finishing this image

    # Write DataFrame
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    df.to_csv(csv_out, index=False)
    return csv_out


# ---------------------------
# Example usage (adapt to your project)
# ---------------------------
if __name__ == "__main__":
    """
    # Example (pseudo-code):

    model = ...  # your trained segmentation model
    state = torch.load("checkpoint.pt", map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    dataloader = ...  # yields (images[N,C,H,W], labels[N,H,W]) or (images, labels, {"path": [...]})

    # Choose the layer to hook:
    # Either pass the module directly:
    # layer = model.decoder.final_block
    # Or pass a dotted path:
    layer_path = "decoder.final_block"

    out_csv = collect_feature_samples(
        model=model,
        dataloader=dataloader,
        feature_layer_path=layer_path,   # or feature_layer=layer
        samples_per_image=128,
        fm_prefix="z",
        csv_out="runs/feat_samples.csv",
        include_pred=True,
        ignore_index=255,                 # or None
        rng_seed=42,
        add_coords=True,
        base_seed=13579,                  # controls per-image selection
        id_from="meta:path",              # or "global_index" (requires DataLoader shuffle=False)
    )
    print("Wrote:", out_csv)
    """
    pass
