import os
from pathlib import Path
import numpy as np
import cv2
from deepface import DeepFace

# ====== 模式A数据目录 ======
MODEA_ROOT = Path(r"D:\KPOP\dataset\images_by_idol")

# ====== 输出向量库 ======
OUT_NPZ = Path(r"D:\KPOP\dataset") / "idol_vectors_opencv.npz"
OUT_EXAMPLES = Path(r"D:\KPOP\dataset") / "idol_examples_modeA.csv"  # 每个idol一张示例图（用于GUI展示）

# ====== 模型/检测器 ======
MODEL_NAME = "Facenet512"
DETECTOR = "opencv"   # 你要opencv
ENFORCE_DETECTION = True

# 可选：最多处理每个idol多少张图（先测试可以设小一点，比如 50）
MAX_PER_IDOL = None  # 或者 200

def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)

def represent_bgr(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = img_bgr[:, :, ::-1]
    reps = DeepFace.represent(
        img_path=img_rgb,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=ENFORCE_DETECTION,
        align=True
    )
    emb = np.array(reps[0]["embedding"], dtype=np.float32)
    return l2norm(emb)

def list_idols_and_images(root: Path):
    idols = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
                imgs.extend(d.glob(ext))
            imgs = sorted(imgs)
            if imgs:
                idols.append((d.name, imgs))
    return idols

def main():
    if not MODEA_ROOT.exists():
        raise FileNotFoundError(f"ModeA folder not found: {MODEA_ROOT}")

    idol_items = list_idols_and_images(MODEA_ROOT)
    if not idol_items:
        raise RuntimeError(f"No idol folders/images found under: {MODEA_ROOT}")

    names = []
    vectors = []
    example_rows = []

    total_imgs = 0
    used_imgs = 0
    skipped = 0

    print(f"Found idols: {len(idol_items)}")
    for idol, img_paths in idol_items:
        if MAX_PER_IDOL is not None:
            img_paths = img_paths[:MAX_PER_IDOL]

        embs = []
        total_imgs += len(img_paths)

        # 选示例图：默认用该idol文件夹第一张
        example_path = str(img_paths[0].resolve())
        example_rows.append((idol, example_path))

        for p in img_paths:
            p = Path(p)
            bgr = cv2.imread(str(p))
            if bgr is None:
                skipped += 1
                continue
            try:
                emb = represent_bgr(bgr)
                embs.append(emb)
                used_imgs += 1
            except Exception:
                # 检测不到脸 / 质量差等
                skipped += 1

        if len(embs) == 0:
            print(f"[WARN] {idol}: no valid faces, skip idol")
            continue

        # idol均值向量（再归一化）
        mean_vec = l2norm(np.mean(np.stack(embs, axis=0), axis=0))

        names.append(idol)
        vectors.append(mean_vec)
        print(f"{idol}: used {len(embs)} imgs")

    names = np.array(names, dtype=object)
    vectors = np.stack(vectors, axis=0).astype(np.float32)

    # 保存 npz
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, names=names, vectors=vectors)
    print(f"\nSaved vectors -> {OUT_NPZ}")
    print(f"Total images scanned: {total_imgs}, used: {used_imgs}, skipped: {skipped}")

    # 保存 examples.csv（给GUI用）
    import csv
    with open(OUT_EXAMPLES, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idol", "example_path"])
        w.writerows(example_rows)
    print(f"Saved examples -> {OUT_EXAMPLES}")

if __name__ == "__main__":
    main()
