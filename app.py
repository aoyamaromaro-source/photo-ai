# app.py

import streamlit as st
from PIL import Image, ImageDraw
import torch
import open_clip
import numpy as np
import io

st.set_page_config(layout="wide")
st.title("📸 AIフォトコンテスト（高速CLIP版）")

# =============================
# UI（最初は軽い）
# =============================
uploaded_files = st.file_uploader("写真を選択（最大10枚）", accept_multiple_files=True)

if uploaded_files and len(uploaded_files) > 10:
    st.warning("10枚までにしてください")
    st.stop()

# =============================
# 実行ボタン
# =============================
run_btn = st.button("AI分析スタート")

# =============================
# 実行
# =============================
if run_btn and uploaded_files:

    with st.spinner("AIモデル読み込み中（初回30秒くらい）..."):

        # ===== モデル（軽量版）
        model, _, preprocess = open_clip.create_model_and_transforms(
            'RN50', pretrained='openai'
        )
        model.eval()

        # ===== カテゴリ
        texts = {
            "dog":["a dog","pet dog"],
            "person":["a person","portrait","face"],
            "landscape":["landscape","nature","scenery"],
            "food":["food","meal","dish"]
        }

        text_features_dict = {}
        for cat, txts in texts.items():
            tokens = open_clip.tokenize(txts)
            with torch.no_grad():
                f = model.encode_text(tokens)
                f /= f.norm(dim=-1, keepdim=True)
            text_features_dict[cat] = f

        # ===== 品質
        quality_texts = [
            "high quality photo",
            "sharp photo",
            "blurry photo",
            "dark photo"
        ]

        tokens = open_clip.tokenize(quality_texts)
        with torch.no_grad():
            qf = model.encode_text(tokens)
            qf /= qf.norm(dim=-1, keepdim=True)

    with st.spinner("画像分析中..."):

        # =============================
        # 画像準備（軽量化）
        # =============================
        image_data = []
        for f in uploaded_files:
            try:
                img = Image.open(f).convert("RGB").resize((224,224))
                image_data.append((f, img))
            except:
                continue

        # =============================
        # バッチ処理（高速化）
        # =============================
        images = torch.cat([
            preprocess(img).unsqueeze(0) for _, img in image_data
        ])

        with torch.no_grad():
            feats = model.encode_image(images)
            feats /= feats.norm(dim=-1, keepdim=True)

        # =============================
        # スコア計算
        # =============================
        results = []

        for i, (file, img) in enumerate(image_data):

            feat = feats[i].unsqueeze(0)

            scores = {}
            for cat, text_feat in text_features_dict.items():
                sims = (feat @ text_feat.T).squeeze()
                scores[cat] = sims.max().item()

            best_cat = max(scores, key=scores.get)

            gray = np.array(img.convert("L"))
            brightness = gray.mean()
            contrast = gray.std()

            quality_score = (feat @ qf.T).squeeze().max().item()

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
                "cat": best_cat
            })

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

        # SNS画像（1位）
        best = top3[0]
        canvas = Image.new("RGB",(1080,1080),(20,20,20))
        canvas.paste(best["img"].resize((1080,1080)),(0,0))

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
        canvas.paste(r["img"].resize((540,540)),positions[i])

    st.image(canvas)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG")

    st.download_button("総合SNS画像DL", buf.getvalue(), "ranking.jpg", "image/jpeg")

# =============================
# 初期メッセージ
# =============================
if not uploaded_files:
    st.info("写真をアップロードして『AI分析スタート』を押してください")
