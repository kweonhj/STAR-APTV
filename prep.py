# prep.py
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff

def load_tif_safe(image_path: Path) -> np.ndarray:
    try:
        image = tiff.imread(str(image_path))
    except Exception as e:
        print(f"[tifffile failed] {image_path.name}: {e}")
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read TIFF file: {image_path}")
    return image


def to_single_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image[..., 0]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    max_val = float(image.max())

    if max_val <= 0:
        return np.zeros_like(image, dtype=np.uint8)

    image_u8 = (image / max_val) * 255.0
    image_u8 = np.clip(image_u8, 0, 255).astype(np.uint8)
    return image_u8


def convert_tif_to_png_renamed(input_dir, output_dir) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    if not tif_files:
        print(f"No TIFF files found in: {input_dir}")
        return

    for i, image_path in enumerate(tif_files, start=1):
        try:
            print(f"Processing: {image_path.name}")

            image = load_tif_safe(image_path)
            image = to_single_channel(image)
            image_u8 = normalize_to_uint8(image)

            output_name = f"{i:04d}.png"
            output_path = output_dir / output_name

            ok = cv2.imwrite(str(output_path), image_u8)
            if not ok:
                raise IOError(f"Failed to save image: {output_path}")

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Failed to process {image_path.name}: {e}")

    print("All TIFF images were converted to PNG successfully.")


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def to_single_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image[..., 0]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def process_one_image(
    image_path: Path,
    output_dir: Path,
    gamma: float,
    output_name: str,
) -> Path:
    print(f"Processing: {image_path.name}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read PNG file: {image_path}")

    image = to_single_channel(image)

    print(
        f"Loaded image: min={image.min()}, "
        f"max={image.max()}, dtype={image.dtype}, shape={image.shape}"
    )

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        max_val = float(image.max())
        if max_val > 0:
            image = (image / max_val) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    corrected = adjust_gamma(image, gamma=gamma)

    print(
        f"After gamma correction: min={corrected.min()}, "
        f"max={corrected.max()}, dtype={corrected.dtype}"
    )

    output_path = output_dir / output_name
    ok = cv2.imwrite(str(output_path), corrected)
    if not ok:
        raise IOError(f"Failed to save image: {output_path}")

    print(f"Saved: {output_path}")
    return output_path


def run_prep(
    input_dir,
    output_dir,
    gamma_value: float = 1.0,
    keep_stem: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No .png files found in: {input_dir}")
        return

    for i, image_path in enumerate(png_files, start=1):
        try:
            if keep_stem:
                output_name = f"{image_path.stem}.png"
            else:
                output_name = f"{i:04d}.png"

            process_one_image(
                image_path=image_path,
                output_dir=output_dir,
                gamma=gamma_value,
                output_name=output_name,
            )
        except Exception as e:
            print(f"Failed to process {image_path.name}: {e}")

    print("Gamma correction applied and PNG images saved successfully.")
