# app.py

import streamlit as st
from PIL import Image, ImageDraw
import torch
import open_clip
import numpy as np
from pillow_heif import register_heif_opener
import io

register_heif_opener()
st.set_page_config(layout="wide")

st.title("📸 AIフォトコンテスト（CLIP版完成形）")

# =============================
# モデルロード（キャッシュ）
# =============================
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    model.eval()
    return model, preprocess

model, preprocess = load_model()

# =============================
# カテゴリ定義
# =============================
texts = {
    "dog":[
        "a dog","cute dog","pet dog"
    ],
    "person":[
        "a person","people","family","friends","portrait","face"
    ],
    "landscape":[
        "landscape","nature","mountain","outdoor","sky","scenery"
    ],
    "food":[
        "food","meal","dish","delicious food","restaurant food"
    ]
}

# =============================
# テキスト特徴（キャッシュ）
# =============================
@st.cache_resource
def build_text_features():
    text_features_dict = {}
    for cat, txts in texts.items():
        tokens = open_clip.tokenize(txts)
        with torch.no_grad():
            f = model.encode_text(tokens)
            f /= f.norm(dim=-1, keepdim=True)
        text_features_dict[cat] = f
    return text_features_dict

text_features_dict = build_text_features()

# =============================
# 品質テキスト
# =============================
@st.cache_resource
def build_quality():
    quality_texts = [
        "high quality photo",
        "sharp photo",
        "well composed photo",
        "blurry photo",
        "dark photo"
    ]
    tokens = open_clip.tokenize(quality_texts)
    with torch.no_grad():
        qf = model.encode_text(tokens)
        qf /= qf.norm(dim=-1, keepdim=True)
    return qf

quality_feature = build_quality()

# =============================
# スコア関数
# =============================
def get_best_score(feat, text_features):
    sims = (feat @ text_features.T).squeeze()
    return sims.max().item()

# =============================
# 推論（キャッシュ）
# =============================
@st.cache_data
def run_inference(image_data):

    results = []

    for file, img in image_data:

        image = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            feat = model.encode_image(image)
            feat /= feat.norm(dim=-1, keepdim=True)

        scores = {}
        for cat, text_feat in text_features_dict.items():
            scores[cat] = get_best_score(feat, text_feat)

        best_cat = max(scores, key=scores.get)

        gray = np.array(img.convert("L"))
        brightness = gray.mean()
        contrast = gray.std()

        quality_score = get_best_score(feat, quality_feature)

        total = (
            scores[best_cat]*100*0.5 +
            quality_score*100*0.3 +
            (100-abs(brightness-120))*0.1 +
            min(100, contrast*2)*0.1
        )

        results.append({
            "file": file,
            "img": img,
            "total": total,
            "cat": best_cat,
            "scores": scores
        })

    return results

# =============================
# アップロード
# =============================
uploaded_files = st.file_uploader("写真を選択", accept_multiple_files=True)

if uploaded_files:

    image_data = []
    for f in uploaded_files:
        try:
            img = Image.open(f).convert("RGB").resize((384,384))
            image_data.append((f,img))
        except:
            continue

    if st.button("ランキング実行"):
        with st.spinner("AI分析中..."):
            results = run_inference(image_data)

        # =============================
        # 総合ランキング
        # =============================
        results_sorted = sorted(results, key=lambda x: x["total"], reverse=True)

        st.header("🏆 総合ランキング")

        for i, r in enumerate(results_sorted[:5]):
            col1, col2 = st.columns([1,1])

            with col1:
                st.image(r["img"], use_container_width=True)

            with col2:
                st.subheader(f"{i+1}位")
                st.metric("スコア", round(r["total"],1))
                st.write(f"カテゴリ：{r['cat']}")

        # =============================
        # ジャンル別TOP3
        # =============================
        st.header("🎯 ジャンル別ランキング")

        categories = ["dog","person","landscape","food"]

        for cat in categories:

            st.subheader(cat)

            filtered = [r for r in results if r["cat"] == cat]

            if len(filtered) == 0:
                st.write("該当なし")
                continue

            top3 = sorted(filtered, key=lambda r: r["total"], reverse=True)[:3]

            cols = st.columns(len(top3))

            for i, r in enumerate(top3):
                with cols[i]:
                    st.image(r["img"], use_container_width=True)
                    st.write(f"{i+1}位")
                    st.write(f"{round(r['total'],1)}点")

            # =============================
            # SNS画像（ジャンル1位）
            # =============================
            best = top3[0]

            canvas = Image.new("RGB",(1080,1080),(20,20,20))
            img_resized = best["img"].resize((1080,1080))
            canvas.paste(img_resized,(0,0))

            draw = ImageDraw.Draw(canvas)
            draw.text((30,30), f"{cat} BEST1", fill=(255,255,255))

            buf = io.BytesIO()
            canvas.save(buf, format="JPEG")

            st.download_button(
                f"{cat} SNS画像DL",
                buf.getvalue(),
                file_name=f"{cat}_best.jpg",
                mime="image/jpeg"
            )

        # =============================
        # 総合SNS（TOP4）
        # =============================
        st.header("📱 総合SNS画像")

        canvas = Image.new("RGB",(1080,1080),(10,10,10))
        positions = [(0,0),(540,0),(0,540),(540,540)]

        for i,r in enumerate(results_sorted[:4]):
            img = r["img"].resize((540,540))
            canvas.paste(img,positions[i])

        st.image(canvas)

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG")

        st.download_button("総合SNS画像DL", buf.getvalue(), "ranking.jpg", "image/jpeg")
