# cal_sam2.py

import os
import copy
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_closing
from skimage.morphology import remove_small_objects

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


SAM2_CHECKPOINT = "/data1/jaejung/others/hyeokjin/sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def get_device(device_str=None):
    if device_str is not None:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda:1")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.float16).__enter__()
        gpu_index = device.index if device.index is not None else 0
        if torch.cuda.get_device_properties(gpu_index).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS."
        )

    return device


def clean_segmentation(mask, min_size=50):
    return remove_small_objects(mask, min_size=min_size)


def show_anns(image, anns, output_path, borders=True, dpi=300):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0

    for index, ann in enumerate(sorted_anns):
        m = ann["segmentation"]
        cleaned = clean_segmentation(m, min_size=50)
        closed = binary_closing(cleaned, structure=np.ones((3, 3)))

        sorted_anns[index]["segmentation"] = closed

        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask

        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    return sorted_anns


def merge_masks_with_colors(masks):
    num_masks, height, width = masks.shape
    merged_image = np.zeros((height, width, 3), dtype=np.uint8)

    colors = plt.cm.jet(np.linspace(0, 1, num_masks))[:, :3] * 255

    for i, mask in enumerate(masks):
        merged_image[mask > 0] = colors[i]

    return merged_image


def _shift_image(img, dx, dy):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _shift_mask_bool(mask, dx, dy):
    h, w = mask.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(
        mask.astype(np.uint8),
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out.astype(bool)


def generate_phase_locked(mask_generator, image_f32_rgb, pad_border=16, phase_ratio=0.5):
    if pad_border > 0:
        img_pad = cv2.copyMakeBorder(
            (image_f32_rgb * 255).astype(np.uint8),
            pad_border,
            pad_border,
            pad_border,
            pad_border,
            cv2.BORDER_REFLECT_101,
        ).astype(np.float32) / 255.0
    else:
        img_pad = image_f32_rgb.copy()

    hp, wp = img_pad.shape[:2]
    points_per_side = getattr(mask_generator, "points_per_side", 128)
    cell = max(hp, wp) / max(points_per_side, 1)

    dx = int(round(phase_ratio * cell))
    dy = int(round(phase_ratio * cell))
    img_shift = _shift_image(img_pad, dx, dy)

    anns = mask_generator.generate(img_shift)

    outs = []
    for d in anns:
        seg = d["segmentation"] > 0
        seg_un = _shift_mask_bool(seg, -dx, -dy)

        if pad_border > 0:
            seg_un = seg_un[pad_border:-pad_border, pad_border:-pad_border]

        nd = copy.copy(d)
        nd["segmentation"] = seg_un
        nd["area"] = int(seg_un.sum())
        outs.append(nd)

    return outs


def build_sam2_model(device):
    return build_sam2(
        MODEL_CFG,
        SAM2_CHECKPOINT,
        device=device,
        apply_postprocessing=False,
    )


def build_mask_generator(sam2_model):
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=16,
        points_per_batch=16,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.15,
        crop_n_points_downscale_factor=0.5,
        min_mask_region_area=100.0,
        use_m2m=True,
    )


def process_one_image(image_path, output_dir, mask_generator):
    start_time = time()

    file_name = os.path.basename(image_path)
    output_file = f"processed_{file_name}"
    output_path = os.path.join(output_dir, output_file)

    image = Image.open(image_path)
    image = image.convert("L")
    image = np.array(image)

    image = image / 255.0
    image = np.stack([image] * 3, axis=-1)
    image = np.array(image, dtype=np.float32)

    masks = generate_phase_locked(mask_generator, image, pad_border=16, phase_ratio=0.5)

    end_time = time()
    print("Time: ", end_time - start_time, "s")

    seg_output_file = f"Seg_{file_name.replace('.png', '.npy')}"
    seg_output_path = os.path.join(output_dir, seg_output_file)

    np.save(seg_output_path, masks)
    print(f"Saved filtered masks to {output_path}")

    seg_result = show_anns(image, masks, output_path)

    if seg_result is None or len(seg_result) == 0:
        print(f"No masks found for {file_name}")
        return

    segmentations = np.array([mask["segmentation"] for mask in seg_result])
    merged_image_with_colors = merge_masks_with_colors(segmentations)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(merged_image_with_colors, alpha=0.3)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


def run_sam2(input_dir, output_dir, indices, device_str=None):
    os.makedirs(output_dir, exist_ok=True)

    device = get_device(device_str)
    sam2_model = build_sam2_model(device)
    mask_generator = build_mask_generator(sam2_model)

    for index in indices:
        file_name = f"{index:04d}.png"
        image_path = os.path.join(input_dir, file_name)

        if not os.path.exists(image_path):
            print(f"File {image_path} does not exist.")
            continue

        process_one_image(image_path, output_dir, mask_generator)