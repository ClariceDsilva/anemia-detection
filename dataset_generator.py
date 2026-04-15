"""
dataset_generator.py
Generates a synthetic anemia detection dataset with realistic visual properties.
Anemic conjunctiva/nailbeds are pale/yellowish; normal ones are pink/red.
"""

import os
import numpy as np
import cv2
import json
import random
from pathlib import Path

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

DATASET_DIR = Path("dataset")
CLASSES = ["anemic", "normal"]
SAMPLES_PER_CLASS = 400  # 800 total images
IMG_SIZE = 224


def add_texture(img, intensity=0.3):
    """Add biological texture to image."""
    noise = np.random.normal(0, intensity * 30, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255)
    return img.astype(np.uint8)


def add_veins(img, color, count=5):
    """Add vein-like structures."""
    for _ in range(count):
        x1 = random.randint(20, IMG_SIZE - 20)
        y1 = random.randint(20, IMG_SIZE - 20)
        x2 = random.randint(20, IMG_SIZE - 20)
        y2 = random.randint(20, IMG_SIZE - 20)
        thickness = random.randint(1, 3)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def generate_anemic_image(idx):
    """
    Generate synthetic anemic conjunctiva/nail image.
    Characteristics: pale, yellowish-white, low saturation.
    """
    # Base pale color (low hemoglobin → pale pink/white/yellow)
    r = random.randint(200, 240)
    g = random.randint(175, 215)
    b = random.randint(165, 205)

    img = np.full((IMG_SIZE, IMG_SIZE, 3), [b, g, r], dtype=np.uint8)

    # Add gradient for depth
    for i in range(IMG_SIZE):
        factor = 1.0 - (i / IMG_SIZE) * 0.15
        img[i, :] = np.clip(img[i, :].astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Add pale vein structures (barely visible)
    vein_color = (max(0, b - 25), max(0, g - 20), max(0, r - 15))
    img = add_veins(img, vein_color, count=random.randint(3, 7))

    # Add elliptical conjunctiva/nail shape
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    axes = (random.randint(80, 100), random.randint(60, 80))
    cv2.ellipse(img, center, axes, random.randint(0, 30), 0, 360,
                (max(0, b - 15), max(0, g - 10), max(0, r - 10)), -1)

    # Add texture
    img = add_texture(img, intensity=0.2)

    # Slight blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Add slight yellowish tint (jaundice/pallor)
    yellow_overlay = np.zeros_like(img)
    yellow_overlay[:, :] = [0, 15, 20]  # BGR slight yellow
    img = np.clip(img.astype(np.float32) + yellow_overlay.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)

    return img


def generate_normal_image(idx):
    """
    Generate synthetic normal conjunctiva/nail image.
    Characteristics: healthy pink/red, well-vascularized.
    """
    # Base healthy pink-red color (good hemoglobin → rich pink)
    r = random.randint(200, 245)
    g = random.randint(100, 150)
    b = random.randint(110, 155)

    img = np.full((IMG_SIZE, IMG_SIZE, 3), [b, g, r], dtype=np.uint8)

    # Add gradient for depth
    for i in range(IMG_SIZE):
        factor = 1.0 - (i / IMG_SIZE) * 0.2
        img[i, :] = np.clip(img[i, :].astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Add visible red vein structures
    vein_color = (max(0, b - 40), max(0, g - 30), min(255, r + 10))
    img = add_veins(img, vein_color, count=random.randint(5, 10))

    # Add elliptical conjunctiva/nail shape
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    axes = (random.randint(80, 100), random.randint(60, 80))
    cv2.ellipse(img, center, axes, random.randint(0, 30), 0, 360,
                (max(0, b - 20), max(0, g - 15), min(255, r + 5)), -1)

    # Add texture
    img = add_texture(img, intensity=0.15)

    # Slight blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def augment_image(img):
    """Apply data augmentation."""
    augmented = []

    # Original
    augmented.append(img)

    # Horizontal flip
    augmented.append(cv2.flip(img, 1))

    # Brightness variations
    for factor in [0.8, 1.2]:
        bright = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        augmented.append(bright)

    # Rotation
    for angle in [10, -10, 20, -20]:
        M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
        augmented.append(rotated)

    # Slight color jitter
    jittered = img.copy().astype(np.float32)
    for c in range(3):
        jittered[:, :, c] = np.clip(jittered[:, :, c] + random.uniform(-20, 20), 0, 255)
    augmented.append(jittered.astype(np.uint8))

    return augmented


def generate_dataset():
    """Generate complete dataset."""
    print("=" * 60)
    print("ANEMIA DETECTION - DATASET GENERATOR")
    print("=" * 60)

    stats = {}

    for cls in CLASSES:
        cls_dir = DATASET_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating '{cls}' images...")
        count = 0

        for idx in range(SAMPLES_PER_CLASS):
            if cls == "anemic":
                img = generate_anemic_image(idx)
            else:
                img = generate_normal_image(idx)

            # Save original
            fname = cls_dir / f"{cls}_{idx:04d}.jpg"
            cv2.imwrite(str(fname), img)
            count += 1

            # Augment every 4th image to add variety
            if idx % 4 == 0:
                aug_imgs = augment_image(img)
                for aug_idx, aug_img in enumerate(aug_imgs[1:], 1):  # skip original
                    aug_fname = cls_dir / f"{cls}_{idx:04d}_aug{aug_idx}.jpg"
                    cv2.imwrite(str(aug_fname), aug_img)
                    count += 1

            if (idx + 1) % 100 == 0:
                print(f"  [{cls}] Generated {idx + 1}/{SAMPLES_PER_CLASS} base images...")

        stats[cls] = count
        print(f"  [{cls}] Total: {count} images saved.")

    # Save dataset info
    info = {
        "classes": CLASSES,
        "samples": stats,
        "total": sum(stats.values()),
        "image_size": IMG_SIZE,
        "description": "Synthetic anemia detection dataset based on conjunctiva/nail pallor"
    }
    with open(DATASET_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Dataset generated: {sum(stats.values())} total images")
    print(f"  Anemic: {stats['anemic']}, Normal: {stats['normal']}")
    print(f"  Saved to: {DATASET_DIR.resolve()}")
    print(f"{'=' * 60}")

    return info


if __name__ == "__main__":
    generate_dataset()
