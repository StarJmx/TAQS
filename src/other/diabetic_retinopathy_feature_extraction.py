import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import convolve
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import frangi
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_image_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def brightness_standardize_bgr(bgr_u8: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    l_eq = cv2.normalize(l_eq, None, 0, 255, cv2.NORM_MINMAX)
    lab_eq = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def get_retina_mask(gray_u8: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray_u8, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask > 0


def count_connected_components(binary_mask: np.ndarray, min_area: int, max_area: Optional[int] = None) -> int:
    labeled = label(binary_mask)
    count = 0
    for region in regionprops(labeled):
        area = region.area
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        count += 1
    return count


def sum_connected_component_area(binary_mask: np.ndarray, min_area: int) -> float:
    labeled = label(binary_mask)
    area_sum = 0.0
    for region in regionprops(labeled):
        if region.area >= min_area:
            area_sum += float(region.area)
    return area_sum


def extract_lesion_features(green_u8: np.ndarray, retina_mask: np.ndarray) -> Dict[str, float]:
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    exudate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    micro_candidate = cv2.morphologyEx(green_u8, cv2.MORPH_BLACKHAT, small_kernel)
    _, micro_bin = cv2.threshold(micro_candidate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    micro_bin = (micro_bin > 0) & retina_mask
    num_microaneurysms = count_connected_components(micro_bin, min_area=3, max_area=80)

    hemo_candidate = cv2.morphologyEx(green_u8, cv2.MORPH_BLACKHAT, large_kernel)
    _, hemo_bin = cv2.threshold(hemo_candidate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hemo_bin = (hemo_bin > 0) & retina_mask
    area_hemorrhages = sum_connected_component_area(hemo_bin, min_area=80)

    exudate_candidate = cv2.morphologyEx(green_u8, cv2.MORPH_TOPHAT, exudate_kernel)
    _, exudate_bin = cv2.threshold(exudate_candidate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    exudate_bin = (exudate_bin > 0) & retina_mask
    num_exudates = count_connected_components(exudate_bin, min_area=5)
    area_exudates = sum_connected_component_area(exudate_bin, min_area=5)

    return {
        "num_microaneurysms": float(num_microaneurysms),
        "area_hemorrhages": float(area_hemorrhages),
        "area_exudates": float(area_exudates),
        "num_exudates": float(num_exudates),
    }


def fractal_dimension_boxcount(binary_mask: np.ndarray) -> float:
    image = binary_mask.astype(bool)
    if image.sum() == 0:
        return 0.0
    h, w = image.shape
    p = min(h, w)
    max_power = int(np.floor(np.log2(p)))
    if max_power < 2:
        return 0.0

    sizes = 2 ** np.arange(max_power, 1, -1)
    counts: List[int] = []
    valid_sizes: List[int] = []

    for size in sizes:
        h_trim = (h // size) * size
        w_trim = (w // size) * size
        if h_trim == 0 or w_trim == 0:
            continue
        trimmed = image[:h_trim, :w_trim]
        blocks = trimmed.reshape(h_trim // size, size, w_trim // size, size)
        occupied = blocks.any(axis=(1, 3))
        count = int(np.count_nonzero(occupied))
        if count > 0:
            counts.append(count)
            valid_sizes.append(int(size))

    if len(counts) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(1.0 / np.array(valid_sizes)), np.log(np.array(counts)), 1)
    return float(coeffs[0])


def extract_vessel_features(green_u8: np.ndarray, retina_mask: np.ndarray) -> Dict[str, float]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_eq = clahe.apply(green_u8)
    green_float = green_eq.astype(np.float32) / 255.0

    vessel_prob = frangi(green_float)
    vessel_prob = np.nan_to_num(vessel_prob, nan=0.0, posinf=0.0, neginf=0.0)
    vessel_prob = (vessel_prob - vessel_prob.min()) / (vessel_prob.max() - vessel_prob.min() + 1e-8)
    vessel_u8 = (vessel_prob * 255).astype(np.uint8)

    _, vessel_bin_u8 = cv2.threshold(vessel_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vessel_bin = (vessel_bin_u8 > 0) & retina_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_bin = cv2.morphologyEx(vessel_bin.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    vessel_density = float(vessel_bin.sum() / (retina_mask.sum() + 1e-8))
    fractal_dimension = fractal_dimension_boxcount(vessel_bin)

    skeleton = skeletonize(vessel_bin)
    neighbors = convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0)
    neighbors = neighbors - skeleton.astype(np.uint8)
    branch_points = skeleton & (neighbors >= 3)
    num_branch_points = float(np.count_nonzero(branch_points))

    comp_label = label(skeleton)
    tortuosities: List[float] = []
    for region in regionprops(comp_label):
        if region.area < 10:
            continue
        minr, minc, maxr, maxc = region.bbox
        diag = np.hypot(maxr - minr, maxc - minc)
        if diag > 0:
            tortuosities.append(float(region.area / diag))
    vessel_tortuosity = float(np.mean(tortuosities)) if tortuosities else 1.0

    return {
        "vessel_tortuosity": vessel_tortuosity,
        "vessel_density": vessel_density,
        "fractal_dimension": float(fractal_dimension),
        "num_branch_points": num_branch_points,
    }


def extract_texture_features(gray_u8: np.ndarray, retina_mask: np.ndarray) -> Dict[str, float]:
    quant = (gray_u8 // 16).astype(np.uint8)
    glcm = graycomatrix(
        quant,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True,
    )
    glcm_contrast = float(graycoprops(glcm, "contrast").mean())
    glcm_energy = float(graycoprops(glcm, "energy").mean())
    glcm_homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    glcm_correlation = float(graycoprops(glcm, "correlation").mean())

    lap = cv2.Laplacian(gray_u8, cv2.CV_32F)
    texture_contrast = float(np.abs(lap[retina_mask]).mean()) if retina_mask.any() else float(np.abs(lap).mean())

    return {
        "texture_contrast": texture_contrast,
        "glcm_contrast": glcm_contrast,
        "glcm_energy": glcm_energy,
        "glcm_homogeneity": glcm_homogeneity,
        "glcm_correlation": glcm_correlation,
    }


def extract_color_global_features(rgb_u8: np.ndarray, gray_u8: np.ndarray, retina_mask: np.ndarray) -> Dict[str, float]:
    valid = retina_mask if retina_mask.any() else np.ones_like(retina_mask, dtype=bool)
    r = rgb_u8[:, :, 0][valid].astype(np.float32)
    g = rgb_u8[:, :, 1][valid].astype(np.float32)
    b = rgb_u8[:, :, 2][valid].astype(np.float32)
    gray = gray_u8[valid].astype(np.float32)

    mean_r = float(r.mean())
    mean_g = float(g.mean())
    mean_b = float(b.mean())
    std_r = float(r.std())
    std_g = float(g.std())
    std_b = float(b.std())

    return {
        "color_mean_R": mean_r,
        "color_mean_G": mean_g,
        "color_mean_B": mean_b,
        "mean_R": mean_r,
        "mean_G": mean_g,
        "mean_B": mean_b,
        "std_R": std_r,
        "std_G": std_g,
        "std_B": std_b,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def build_resnet18_encoder(device: torch.device) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        from torchvision import models

        base = models.resnet18(pretrained=True)

    encoder = nn.Sequential(*list(base.children())[:-1])
    encoder.to(device)
    encoder.eval()
    return encoder


def extract_resnet18_embedding(rgb_u8: np.ndarray, encoder: nn.Module, device: torch.device) -> np.ndarray:
    rgb_resized = cv2.resize(rgb_u8, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb_resized).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    x = (x - mean) / std
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        feat = encoder(x)

    return feat.flatten().detach().cpu().numpy().astype(np.float32)


def extract_single_image_features(image_path: Path, encoder: nn.Module, device: torch.device) -> Tuple[Dict[str, float], np.ndarray]:
    bgr = load_image_bgr(image_path)
    bgr_std = brightness_standardize_bgr(bgr)
    rgb = cv2.cvtColor(bgr_std, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr_std, cv2.COLOR_BGR2GRAY)
    green = rgb[:, :, 1]
    mask = get_retina_mask(gray)

    features: Dict[str, float] = {}
    features.update(extract_lesion_features(green, mask))
    features.update(extract_vessel_features(green, mask))
    features.update(extract_texture_features(gray, mask))
    features.update(extract_color_global_features(rgb, gray, mask))

    embedding_512 = extract_resnet18_embedding(rgb, encoder, device)
    return features, embedding_512


def run_extraction(
    label_csv: Path,
    images_dir: Path,
    output_csv: Path,
    limit: Optional[int] = None,
    pca_components: int = 50,
    device: str = "cpu",
) -> None:
    df = pd.read_csv(label_csv)
    required = {"file_id", "DR level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Label CSV missing required columns: {missing}")

    work_df = df.head(limit).copy() if limit else df.copy()
    device_torch = torch.device(device)
    encoder = build_resnet18_encoder(device_torch)

    rows = []
    embedding_vectors: List[np.ndarray] = []
    total = len(work_df)

    for _, row in work_df.iterrows():
        file_id = str(row["file_id"]).zfill(6)
        image_path = images_dir / f"{file_id}.tiff"
        if not image_path.exists():
            print(f"[WARN] Missing image: {image_path}")
            continue

        try:
            feat, embedding = extract_single_image_features(image_path, encoder, device_torch)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed on {file_id}: {exc}")
            continue

        feat["file_id"] = file_id
        feat["DR level"] = row["DR level"]
        rows.append(feat)
        embedding_vectors.append(embedding)

        done = len(rows)
        if done % 50 == 0 or done == total:
            print(f"Processed {done}/{total}")

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid samples were processed. Please check image paths and data quality.")

    embedding_matrix = np.vstack(embedding_vectors)
    max_components = min(pca_components, embedding_matrix.shape[0], embedding_matrix.shape[1])
    if max_components < 1:
        raise ValueError("PCA components must be >= 1.")
    if max_components != pca_components:
        print(f"[INFO] Adjust PCA components from {pca_components} to {max_components} due to sample/feature limits.")

    pca = PCA(n_components=max_components, random_state=42)
    pca_matrix = pca.fit_transform(embedding_matrix)
    pca_cols = [f"cnn_pca_{idx:03d}" for idx in range(max_components)]
    pca_df = pd.DataFrame(pca_matrix, columns=pca_cols)

    out = pd.concat([out.reset_index(drop=True), pca_df], axis=1)

    ordered_cols = [
        "file_id",
        "DR level",
        "num_microaneurysms",
        "area_hemorrhages",
        "area_exudates",
        "num_exudates",
        "vessel_tortuosity",
        "vessel_density",
        "fractal_dimension",
        "num_branch_points",
        "texture_contrast",
        "glcm_contrast",
        "glcm_energy",
        "glcm_homogeneity",
        "glcm_correlation",
        "color_mean_R",
        "color_mean_G",
        "color_mean_B",
        "mean_R",
        "mean_G",
        "mean_B",
        "std_R",
        "std_G",
        "std_B",
        "brightness",
        "contrast",
    ]
    ordered_cols.extend(pca_cols)
    out = out[ordered_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    print(f"Rows: {len(out)}, Cols: {len(out.columns)}")
    print(f"PCA components used: {max_components}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DR handcrafted + ResNet18 embedding features to CSV.")
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "_Diabetic Retinopathy.csv",
        help="CSV containing file_id and DR level.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "images",
        help="Directory with .tiff retinal images.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "Diabetic Retinopathy_features_norm.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample size for quick run.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components from 512-dim ResNet18 embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for ResNet18 embedding extraction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(
        label_csv=args.label_csv,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        limit=args.limit,
        pca_components=args.pca_components,
        device=args.device,
    )
