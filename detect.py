import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from deepface import DeepFace


# ================== 你需要改的路径（只改这里） ==================
MODEA_DIR = Path(r"D:\KPOP\dataset\images_by_idol")  # 模式A根目录
VEC_FILE = Path(r"D:\KPOP\dataset\idol_vectors_opencv.npz")        # 你的向量库（建库脚本输出）
# ================================================================


# ================== 模型设置 ==================
MODEL_NAME = "Facenet512"
DETECTOR_FAST = "opencv"       # 快
DETECTOR_FALLBACK = "retinaface"  # opencv失败就兜底
TOPK = 3
# =============================================


# ================== 百分制映射（根据你元英的分数调过） ==================
# 你给的相似度 0.766/0.757/0.732 属于很高的区间
# 映射：0.45->0%，0.80->100%（更符合直觉）
SCORE_LO = 0.20
SCORE_HI = 0.80
# ======================================================================


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


def score_percent(sim: float, lo=SCORE_LO, hi=SCORE_HI) -> int:
    x = (sim - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    # 平滑一下，显示更“像打分”
    x = x * x * (3 - 2 * x)  # smoothstep
    return int(round(x * 100))


def bgr_to_tk(img_bgr, max_w=320, max_h=320) -> ImageTk.PhotoImage:
    if img_bgr is None:
        img_bgr = np.zeros((max_h, max_w, 3), dtype=np.uint8)

    h, w = img_bgr.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img_resized = cv2.resize(img_bgr, (new_w, new_h))
    img_rgb = img_resized[:, :, ::-1]
    pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil)


def get_embedding_with_fallback(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = img_bgr[:, :, ::-1]

    # 先用 opencv（快）
    try:
        reps = DeepFace.represent(
            img_path=img_rgb,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_FAST,
            enforce_detection=True,
            align=True
        )
        emb = np.array(reps[0]["embedding"], dtype=np.float32)
        return l2norm(emb)
    except Exception:
        pass

    # fallback：retinaface（稳）
    reps = DeepFace.represent(
        img_path=img_rgb,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_FALLBACK,
        enforce_detection=True,
        align=True
    )
    emb = np.array(reps[0]["embedding"], dtype=np.float32)
    return l2norm(emb)


def build_example_map_from_modeA(modea_dir: Path) -> dict:
    """
    modeA: images_by_idol/<idol>/*.jpg
    取每个 idol 文件夹里的第一张图片作为示例图
    """
    ex = {}
    if not modea_dir.exists():
        raise FileNotFoundError(f"ModeA folder not found: {modea_dir}")

    for idol_dir in sorted(modea_dir.iterdir()):
        if not idol_dir.is_dir():
            continue

        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            imgs.extend(sorted(idol_dir.glob(ext)))

        if imgs:
            ex[idol_dir.name] = str(imgs[0].resolve())

    if not ex:
        raise RuntimeError(f"No idol images found under: {modea_dir}")

    return ex


class IdolMatchGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Idol Match (Mode A) - Top3")

        # 检查文件
        if not MODEA_DIR.exists():
            messagebox.showerror("Error", f"找不到模式A目录：\n{MODEA_DIR}")
            raise SystemExit(1)
        if not VEC_FILE.exists():
            messagebox.showerror("Error", f"找不到向量库：\n{VEC_FILE}\n请先跑建库脚本生成 npz")
            raise SystemExit(1)

        # 加载向量库
        vec = np.load(VEC_FILE, allow_pickle=True)
        self.names = vec["names"]
        self.vectors = vec["vectors"].astype(np.float32)

        # 示例图
        self.example_map = build_example_map_from_modeA(MODEA_DIR)

        # UI 顶部
        top = tk.Frame(root)
        top.pack(padx=10, pady=10, fill="x")

        self.btn = tk.Button(top, text="选择图片 → 匹配 Top-3", command=self.select_and_match, height=2)
        self.btn.pack(side="left")

        self.status = tk.Label(
            top,
            text=f"就绪",
            anchor="w"
        )
        self.status.pack(side="left", padx=10, fill="x", expand=True)

        # UI 中部
        mid = tk.Frame(root)
        mid.pack(padx=10, pady=5)

        left = tk.LabelFrame(mid, text="输入照片")
        left.grid(row=0, column=0, padx=8, pady=5)
        self.query_img_label = tk.Label(left)
        self.query_img_label.pack(padx=8, pady=8)

        right = tk.LabelFrame(mid, text="")
        right.grid(row=0, column=1, padx=8, pady=5, sticky="n")


        self.result_rows = []
        for i in range(TOPK):
            row = tk.Frame(right)
            row.pack(padx=8, pady=6, fill="x")

            img_lbl = tk.Label(row)
            img_lbl.pack(side="left")

            txt_lbl = tk.Label(row, text=f"{i+1}. -", justify="left", anchor="w", width=60)
            txt_lbl.pack(side="left", padx=10)

            self.result_rows.append((img_lbl, txt_lbl))

        # 防止图片引用被回收
        self._tk_refs = []

    def select_and_match(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if not path:
            return

        try:
            self.status.config(text="读取图片...")
            self.root.update_idletasks()

            q_bgr = cv2.imread(path)
            if q_bgr is None:
                raise ValueError("图片读取失败：建议先用英文路径/文件名测试")

            q_tk = bgr_to_tk(q_bgr, 360, 360)
            self.query_img_label.config(image=q_tk)
            self._tk_refs = [q_tk]

            self.status.config(text="提取人脸特征（opencv，失败则retinaface兜底）...")
            self.root.update_idletasks()

            q_emb = get_embedding_with_fallback(q_bgr)

            self.status.config(text="计算相似度并取Top-3...")
            self.root.update_idletasks()

            sims = self.vectors @ q_emb
            idx = np.argsort(-sims)[:TOPK]

            for rank, i in enumerate(idx, 1):
                idol = str(self.names[i])
                sim = float(sims[i])
                pct = score_percent(sim)

                ex_path = self.example_map.get(idol, None)
                ex_bgr = cv2.imread(ex_path) if ex_path and os.path.isfile(ex_path) else None
                ex_tk = bgr_to_tk(ex_bgr, 240, 240)

                img_lbl, txt_lbl = self.result_rows[rank - 1]
                img_lbl.config(image=ex_tk)
                txt_lbl.config(
                    text=(
                        f"{rank}. {idol}\n"
                        f"相似度: {pct}%\n"
                    )
                )

                self._tk_refs.append(ex_tk)

            self.status.config(text="完成 ✅")

        except Exception as e:
            self.status.config(text="出错 ❌")
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = IdolMatchGUI(root)
    root.mainloop()
