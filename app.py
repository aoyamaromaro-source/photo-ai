import streamlit as st
import os
from PIL import Image
import torch
import open_clip
from datetime import datetime
from PIL import ExifTags
import numpy as np
import streamlit as st
import os
from pillow_heif import register_heif_opener
register_heif_opener()

# 全体に余白
st.set_page_config(layout="wide")

# 構図スコア
def calc_composition_score(img):
    gray = np.array(img.convert("L"))

    h, w = gray.shape

    center = gray[h//4:3*h//4, w//4:3*w//4]

    # 外側は平均だけ取る（安全）
    top = gray[:h//4, :]
    bottom = gray[3*h//4:, :]
    left = gray[:, :w//4]
    right = gray[:, 3*w//4:]

    outer_mean = np.mean([
        top.mean(),
        bottom.mean(),
        left.mean(),
        right.mean()
    ])

    center_mean = center.mean()

    diff = abs(center_mean - outer_mean)

    score = 1 - diff / 255

    return max(0, min(score, 1))

#　光スコア
def calc_light_score(img):
    gray = np.array(img.convert("L"))

    brightness = gray.mean()
    contrast = gray.std()

    brightness_score = 1 - abs(brightness - 140) / 140
    contrast_score = contrast / 100

    return max(0, min((brightness_score*0.6 + contrast_score*0.4), 1))


# ===== CLIP =====
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='openai'
)
model.eval()

# ===== UI =====
st.markdown(
    """
    <h1 style='text-align: center;'>📸 AIフォトコンテスト</h1>
    <p style='text-align: center; color: gray;'>あなたの写真をAIがランキング</p>
    """,
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "📸 写真を選択（複数OK）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.markdown("またはフォルダ指定👇")

folder_path = st.text_input("📁 写真フォルダのパスを入力", value="photos")

if not os.path.exists(folder_path):
    st.error("フォルダが存在しません")
    st.stop()

IMAGE_FOLDER = folder_path

# ===== 月リスト自動生成 =====
months = set()

for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
        continue

    path = os.path.join(IMAGE_FOLDER, filename)

    try:
        img = Image.open(path)
        exif = img._getexif()

        date = None
        if exif:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == "DateTimeOriginal":
                    date = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")

        if date is None:
            if not uploaded_files:
                timestamp = os.path.getmtime(path)
                date = datetime.fromtimestamp(timestamp)
            else:
                date = datetime.now()

        month_key = date.strftime("%Y-%m")
        months.add(month_key)

    except:
        continue

# ソート
months = sorted(list(months))

# 月選択（適当に候補）
TARGET_MONTH = st.selectbox("月を選択", months)

if uploaded_files:
    st.info("※アップロード画像は月フィルタなしで評価しています")

# ===== カテゴリ =====
categories = ["dog", "person", "landscape", "food"]

# ===== カテゴリ判定 =====
texts = {
    "dog": [
        "a photo of a dog",
        "a cute dog",
        "a pet dog",
        "a small dog"
    ],
    "person": [
        "a photo of a person",
        "a portrait of a person",
        "a human face",
        "a smiling person",
        "a person standing",
        "a person taking a photo"
    ],
    "landscape": [
        "a landscape photo",
        "nature scenery",
        "mountains and sky",
        "beautiful view",
        "outdoor scenery"
    ],
    "food": [
        "a photo of food",
        "delicious meal",
        "a dish on a plate",
        "cooked food",
        "gourmet meal"
    ]
}
text_features_dict = {}

for cat, txts in texts.items():
    tokens = open_clip.tokenize(txts)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

    # 👇 平均を取る（これが重要）
    text_features_dict[cat] = features.mean(dim=0)

results = {cat: [] for cat in categories}
processed_files = set()

# ===== データ決定（ここ追加）=====
if uploaded_files:
    files = uploaded_files
else:
    files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))
    ]

# ===== 画像処理 =====
for file in files:
    try:
        if uploaded_files:
            img = Image.open(file)
            filename = file.name
        else:
            filename = file
            path = os.path.join(IMAGE_FOLDER, filename)
            img = Image.open(path)
        # 👇ここに入れる！！
        if filename in processed_files:
            continue
        processed_files.add(filename)

        # ===== 日付取得（EXIF）=====
        date = None

        try:
            if hasattr(img, "_getexif") and img._getexif():
                exif = img._getexif()
                for tag, value in exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    if decoded == "DateTimeOriginal":
                        date = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
        except:
            pass

        if date is None:
            timestamp = os.path.getmtime(path)
            date = datetime.fromtimestamp(timestamp)

        month_key = date.strftime("%Y-%m")

        if not uploaded_files:
            if month_key != TARGET_MONTH:
                continue

        # ===== CLIP =====
        image = preprocess(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        scores = {}

        for cat, text_feat in text_features_dict.items():
            score = (image_features @ text_feat.unsqueeze(1)).item()
            scores[cat] = score

        # 一番高いカテゴリを選ぶ
        category = max(scores, key=scores.get)

        # 被写体スコア
        subject = scores[category]
        
        print("分類:", filename, category, subject)

        # ===== スコア（簡易）=====
        light = calc_light_score(img)
        composition = calc_composition_score(img)

        score = light * 0.3 + composition * 0.3 + subject * 0.4

        if uploaded_files:
            results[category].append((file, score, light, composition, subject))
        else:
            results[category].append((filename, score, light, composition, subject))
    except Exception as e:
        print("エラー:", e)
        continue

# ===== 表示 =====
st.markdown(f"## 📅 {TARGET_MONTH} フォトコン結果")

for cat in categories:
    st.markdown(f"### 🏆 {cat.upper()}")

    results[cat].sort(key=lambda x: x[1], reverse=True)
    top_results = results[cat][:3]

    cols = st.columns(3)

    for i, (data, score, light, composition, subject) in enumerate(top_results):
        if uploaded_files:
            img = Image.open(data)
        else:
            path = os.path.join(IMAGE_FOLDER, data)
            img = Image.open(path)

        with cols[i]:
            st.image(img)

            if i == 0:
                st.markdown("🥇 **1位**")
            elif i == 1:
                st.markdown("🥈 **2位**")
            elif i == 2:
                st.markdown("🥉 **3位**")

            st.markdown(f"スコア: {score:.2f}")

            st.markdown(f"""
光: {light:.2f}  
構図: {composition:.2f}  
被写体: {subject:.2f}
""")
