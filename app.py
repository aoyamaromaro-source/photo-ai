import streamlit as st
import os
from PIL import Image
import torch
import open_clip
from datetime import datetime
from PIL import ExifTags
import numpy as np
from pillow_heif import register_heif_opener

register_heif_opener()

st.set_page_config(layout="wide")

# ===== 軽量CLIP（キャッシュ）=====
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    model.eval()
    return model, preprocess

model, preprocess = load_model()

# ===== スコア =====
def calc_composition_score(img):
    gray = np.array(img.convert("L"))
    h, w = gray.shape

    center = gray[h//4:3*h//4, w//4:3*w//4]

    outer_mean = np.mean([
        gray[:h//4, :].mean(),
        gray[3*h//4:, :].mean(),
        gray[:, :w//4].mean(),
        gray[:, 3*w//4:].mean()
    ])

    diff = abs(center.mean() - outer_mean)
    return max(0, min(1 - diff / 255, 1))


def calc_light_score(img):
    gray = np.array(img.convert("L"))
    brightness = gray.mean()
    contrast = gray.std()

    brightness_score = 1 - abs(brightness - 140) / 140
    contrast_score = contrast / 100

    return max(0, min((brightness_score*0.6 + contrast_score*0.4), 1))

# ===== UI =====
st.title("📸 AIフォトコンテスト")

uploaded_files = st.file_uploader(
    "写真を選択（複数OK）",
    type=["jpg", "jpeg", "png", "heic"],
    accept_multiple_files=True
)

# ===== フィルタ =====
filter_type = st.selectbox(
    "📅 フィルタ",
    ["すべて", "直近50枚", "直近100枚", "今月"]
)

# ===== カテゴリ =====
categories = ["dog", "person", "landscape", "food"]

texts = {
    "dog": ["a dog", "a cute dog", "a pet dog"],
    "person": ["a person", "a human face", "a portrait"],
    "landscape": ["landscape", "nature scenery", "mountain"],
    "food": ["food", "meal", "dish"]
}

# ===== テキスト特徴量 =====
text_features_dict = {}
for cat, txts in texts.items():
    tokens = open_clip.tokenize(txts)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    text_features_dict[cat] = features.mean(dim=0)

# ===== 日付取得 =====
def get_date(img):
    try:
        if hasattr(img, "_getexif") and img._getexif():
            exif = img._getexif()
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag, tag) == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except:
        pass
    return datetime.now()

# ===== メイン処理 =====
if uploaded_files:

    if len(uploaded_files) > 100:
        st.warning("最大100枚までにしてください")
        st.stop()

    image_data = []

    for file in uploaded_files:
        try:
            img = Image.open(file).convert("RGB")

            # 軽量化
            img = img.resize((512, 512))

            date = get_date(img)

            image_data.append((file, img, date))
        except:
            continue

    # ===== ソート =====
    image_data.sort(key=lambda x: x[2], reverse=True)

    # ===== フィルタ =====
    if filter_type == "直近50枚":
        image_data = image_data[:50]
    elif filter_type == "直近100枚":
        image_data = image_data[:100]
    elif filter_type == "今月":
        now = datetime.now()
        image_data = [x for x in image_data if x[2].month == now.month]

    results = {cat: [] for cat in categories}

    # ===== 推論 =====
    for file, img, date in image_data:

        image = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        scores = {}
        for cat, text_feat in text_features_dict.items():
            scores[cat] = (image_features @ text_feat.unsqueeze(1)).item()

        category = max(scores, key=scores.get)
        subject = scores[category]

        light = calc_light_score(img)
        composition = calc_composition_score(img)

        total = light*0.3 + composition*0.3 + subject*0.4

        results[category].append((file, total, light, composition, subject))

    # ===== 表示 =====
    for cat in categories:
        st.subheader(f"🏆 {cat}")

        results[cat].sort(key=lambda x: x[1], reverse=True)

        cols = st.columns(3)

        for i, (file, score, light, comp, subj) in enumerate(results[cat][:3]):
            with cols[i]:
                st.image(file)

                st.markdown(f"""
🥇順位: {i+1}  
スコア: {score:.2f}  
光: {light:.2f}  
構図: {comp:.2f}  
被写体: {subj:.2f}
""")
