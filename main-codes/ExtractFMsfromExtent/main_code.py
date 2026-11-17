from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import cv2

# Optional but recommended for georeferenced output
import rasterio as rio


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_4d_input(chip: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Accepts chip as:
      - torch.Tensor: (C,H,W) or (B,C,H,W)
      - np.ndarray:   (C,H,W) or (H,W,C) or (B,C,H,W)
    Returns torch.Tensor: (B,C,H,W)
    """
    if isinstance(chip, np.ndarray):
        x = torch.from_numpy(chip)
    else:
        x = chip

    if x.dtype != torch.float32:
        x = x.float()

    # (H,W,C) -> (C,H,W) if it looks like channels-last image
    if x.ndim == 3 and x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
        x = x.permute(2, 0, 1)

    if x.ndim == 3:
        x = x.unsqueeze(0)

    if x.ndim != 4:
        raise ValueError(f"chip must be 3D or 4D and become (B,C,H,W). Got {tuple(x.shape)}")

    return x


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if module_name not in modules:
        partial = [n for n in modules.keys() if module_name.split(".")[-1] in n][-50:]
        raise KeyError(
            f"Layer '{module_name}' not found in model.named_modules().\n"
            f"Tip: print names via `for n,_ in model.named_modules(): print(n)`.\n"
            f"Some partial matches: {partial}"
        )
    return modules[module_name]


def _normalize_to_uint8(arr2d: np.ndarray, mode: str = "minmax", eps: float = 1e-8) -> np.ndarray:
    x = arr2d.astype(np.float32)

    if mode == "minmax":
        mn, mx = float(np.min(x)), float(np.max(x))
        denom = (mx - mn) if (mx - mn) > eps else eps
        y = (x - mn) / denom
    elif mode == "absmax":
        m = float(np.max(np.abs(x)))
        denom = m if m > eps else eps
        y = np.abs(x) / denom
    else:
        raise ValueError("mode must be 'minmax' or 'absmax'")

    return (y * 255.0).clip(0, 255).astype(np.uint8)


def _candidate_hw_from_input(input_hw: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Candidate output sizes commonly seen in encoder/decoder pipelines.
    We try these to reshape token tensors (B,HW,C) -> (B,H,W,C).
    """
    H, W = input_hw
    cands = []
    for f in [1, 2, 4, 8, 16]:
        h = H // f
        w = W // f
        if h > 0 and w > 0:
            cands.append((h, w))
    return cands


def _standardize_to_nchw(
    out: torch.Tensor,
    expect_channels: Optional[int],
    input_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert common layouts to (B,C,H,W).

    Supported:
      - (B,C,H,W) : unchanged
      - (B,H,W,C) : permute -> (B,C,H,W) (if C matches expect_channels)
      - (B,HW,C)  : reshape -> (B,H,W,C) -> (B,C,H,W) (if possible)
      - (B,C,HW)  : reshape -> (B,C,H,W) (if possible)
    """
    if not torch.is_tensor(out):
        return out

    # --- Case 1: already 4D ---
    if out.ndim == 4:
        if expect_channels is not None:
            if out.shape[1] == expect_channels:
                return out  # NCHW
            if out.shape[-1] == expect_channels:
                return out.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        return out

    # --- Case 2: token format (B,HW,C) or (B,C,HW) ---
    if out.ndim == 3 and expect_channels is not None:
        B = out.shape[0]

        # (B,HW,C)
        if out.shape[2] == expect_channels:
            HW = out.shape[1]

            for (h, w) in _candidate_hw_from_input(input_hw):
                if h * w == HW:
                    out = out.view(B, h, w, expect_channels)          # (B,H,W,C)
                    return out.permute(0, 3, 1, 2).contiguous()       # (B,C,H,W)

            # Fallback: perfect square
            s = int(round(HW ** 0.5))
            if s * s == HW:
                out = out.view(B, s, s, expect_channels)
                return out.permute(0, 3, 1, 2).contiguous()

            return out  # couldn't reshape

        # (B,C,HW)
        if out.shape[1] == expect_channels:
            HW = out.shape[2]
            for (h, w) in _candidate_hw_from_input(input_hw):
                if h * w == HW:
                    return out.view(B, expect_channels, h, w).contiguous()
            s = int(round(HW ** 0.5))
            if s * s == HW:
                return out.view(B, expect_channels, s, s).contiguous()
            return out

    return out


def _read_extent_as_tensor_cv2(image_in: str, n_channels: int) -> torch.Tensor:
    """
    Reads an image with OpenCV and returns (C,H,W) float in [0,1].
    Assumes RGB-like inputs.
    """
    img = cv2.imread(image_in, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_in}")

    # If it’s 3-channel, ensure RGB order
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)

    # If single-channel and user expects 1, keep it
    if img.ndim == 2:
        img = img[:, :, None]

    # Keep only requested channels
    img = img[:, :, :n_channels]

    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return x  # (C,H,W)


@torch.no_grad()
def extract_feature_maps_over_extent(
    model: nn.Module,
    image_in: str,
    layer_path: str,
    out_dir: Union[str, Path],
    chip_size: int = 224,
    stride_x: int = 112,
    stride_y: int = 112,
    crop: int = 16,
    n_channels: int = 3,
    expect_channels: int = 96,
    device: Optional[Union[str, torch.device]] = None,
    # saving
    save_geotiff: bool = True,
    geotiff_name: str = "feature_maps_96.tif",
    save_images: bool = False,
    image_ext: str = "png",
    image_norm: str = "minmax",
    image_prefix: str = "fm",
    # debug
    print_shapes_once: bool = False,
) -> Dict[str, Any]:
    """
    Sliding-window extraction of 96 feature maps across a large extent.

    - Hooks `layer_path` once.
    - For each tile, runs model(tile) and captures that layer output.
    - Standardizes output to NCHW (handles NHWC and token formats).
    - Stitches into a full tensor: (96, H, W) using `crop` like your geoInfer.
    - Optionally writes a 96-band GeoTIFF (float32) and/or per-channel PNGs.

    Assumptions:
      - Hooked layer produces spatial output at the SAME resolution as the tile (typically final decoder block).
      - If it produces lower-res maps, you’d need upsampling before stitching.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    out_dir = _ensure_dir(out_dir)

    # Read full extent
    t_arr = _read_extent_as_tensor_cv2(image_in, n_channels=n_channels)  # (C,H,W)
    H, W = int(t_arr.shape[1]), int(t_arr.shape[2])

    # Output full-extent FM stack (CPU)
    fm_full = torch.zeros((expect_channels, H, W), dtype=torch.float32)

    # Set model eval once
    model.eval()

    # Hook target layer once
    target_module = _get_module_by_name(model, layer_path)
    captured = {"out": None}

    def hook_fn(module, inp, out):
        if torch.is_tensor(out):
            captured["out"] = out.detach()
        elif isinstance(out, (tuple, list)):
            for item in out:
                if torch.is_tensor(item):
                    captured["out"] = item.detach()
                    break
        else:
            captured["out"] = None

    hook_handle = target_module.register_forward_hook(hook_fn)

    try:
        # Sliding windows
        size = chip_size

        for r in range(0, H, stride_y):
            r1 = min(r, H - size)
            r2 = r1 + size

            for c in range(0, W, stride_x):
                c1 = min(c, W - size)
                c2 = c1 + size

                tile = t_arr[0:n_channels, r1:r2, c1:c2]  # (C,size,size)
                tile = tile.to(device).unsqueeze(0)       # (1,C,size,size)

                captured["out"] = None
                _ = model(tile)

                if captured["out"] is None:
                    raise RuntimeError(
                        f"Hook on '{layer_path}' did not capture output during forward pass."
                    )

                out = captured["out"]

                if print_shapes_once:
                    print(f"[hook] raw shape: {tuple(out.shape)}")
                    print_shapes_once = False

                # Standardize to (B,96,size,size) if possible
                out = _standardize_to_nchw(out, expect_channels, input_hw=(size, size))

                if not torch.is_tensor(out) or out.ndim != 4:
                    raise RuntimeError(
                        f"After standardization, expected 4D tensor for '{layer_path}'. "
                        f"Got: type={type(out)} shape={getattr(out, 'shape', None)}"
                    )

                if out.shape[1] != expect_channels:
                    raise RuntimeError(
                        f"Channel mismatch at '{layer_path}': expected {expect_channels}, got {out.shape[1]}"
                    )

                fm_tile = out[0].detach()  # (96, ht, wt)
                
                ht, wt = int(fm_tile.shape[1]), int(fm_tile.shape[2])
                if (ht != size) or (wt != size):
                    # Upsample to chip size (224x224) so stitching works
                    fm_tile = torch.nn.functional.interpolate(
                        fm_tile.unsqueeze(0),          # (1,C,ht,wt)
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)                       # (C,size,size)

                fm_tile = fm_tile.cpu()               # now (96,224,224)


                # Stitch using crop margins (clean seam handling)
                dest_r1 = r1 + (0 if r1 == 0 else crop)
                dest_r2 = r2 - (0 if r2 == H else crop)
                dest_c1 = c1 + (0 if c1 == 0 else crop)
                dest_c2 = c2 - (0 if c2 == W else crop)

                src_r1 = 0 if r1 == 0 else crop
                src_r2 = size if r2 == H else size - crop
                src_c1 = 0 if c1 == 0 else crop
                src_c2 = size if c2 == W else size - crop

                fm_full[:, dest_r1:dest_r2, dest_c1:dest_c2] = fm_tile[:, src_r1:src_r2, src_c1:src_c2]

    finally:
        hook_handle.remove()
        torch.cuda.empty_cache()

    saved: Dict[str, Any] = {}

    # Save as 96-band GeoTIFF if possible
    if save_geotiff:
        try:
            with rio.open(image_in) as src:
                profile = src.profile.copy()

            profile.update(
                driver="GTiff",
                dtype="float32",
                count=expect_channels,
                compress="LZW",
            )

            out_tif = out_dir / geotiff_name
            fm_np = fm_full.numpy().astype(np.float32)

            with rio.open(out_tif, "w", **profile) as dst:
                dst.write(fm_np)  # expects (bands, H, W)

            saved["geotiff"] = str(out_tif)

        except Exception as e:
            saved["geotiff_error"] = f"Could not write GeoTIFF (maybe input not georeferenced): {e}"

    # Optional: save per-channel preview images (NOT georeferenced)
    if save_images:
        ext = image_ext.lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"
        if ext not in ("png", "jpg"):
            raise ValueError("image_ext must be 'png' or 'jpg'/'jpeg'")

        img_dir = _ensure_dir(out_dir / f"channels_{ext}")
        fm_np = fm_full.numpy()

        for i in range(expect_channels):
            img_u8 = _normalize_to_uint8(fm_np[i], mode=image_norm)
            out_path = img_dir / f"{image_prefix}_{i:03d}.{ext}"
            ok = cv2.imwrite(str(out_path), img_u8)
            if not ok:
                raise RuntimeError(f"Failed to write image: {out_path}")

        saved["images_dir"] = str(img_dir)

    return {
        "layer_path": layer_path,
        "extent_shape": (H, W),
        "feature_map_shape": (expect_channels, H, W),
        "saved": saved,
    }
