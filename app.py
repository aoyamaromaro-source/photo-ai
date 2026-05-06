# app.py

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import io

st.set_page_config(page_title="写真ランキングAI", layout="wide")

st.title("📷 写真ランキングAI（完成版）")
st.caption("軽量だけどちゃんとジャンル分け＋ランキング＋SNS画像生成")

# -------------------------
# スコア計算
# -------------------------
def calc_score(img):
    arr = np.array(img)

    brightness = arr.mean()
    contrast = arr.std()
    colorfulness = (
        np.std(arr[:,:,0]) +
        np.std(arr[:,:,1]) +
        np.std(arr[:,:,2])
    )

    randomness = np.random.uniform(0, 15)

    score = brightness * 0.2 + contrast * 0.3 + colorfulness * 0.3 + randomness
    return round(score, 1)

# -------------------------
# カテゴリ判定（画像ベース）
# -------------------------
def detect_category(img):
    arr = np.array(img)

    r = arr[:,:,0].mean()
    g = arr[:,:,1].mean()
    b = arr[:,:,2].mean()

    color_var = arr.std()

    # 風景（緑・青）
    if g > r and g > b:
        return "🌄 風景"

    # 食べ物（暖色）
    if r > 120 and g > 80:
        return "🍜 食べ物"

    # 人物（明るくてコントラスト低め）
    if color_var < 50 and r > 100:
        return "😊 人物"

    # ペット（色が強い）
    if color_var > 70:
        return "🐶 ペット"

    return "📷 その他"

# -------------------------
# SNS画像生成
# -------------------------
def create_sns_image(img, title):
    base = img.copy().resize((600, 600))

    canvas = Image.new("RGB", (600, 700), "white")
    canvas.paste(base, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((20, 620), title, fill="black")

    return canvas

# -------------------------
# アップロード
# -------------------------
files = st.file_uploader(
    "写真をアップロード",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if files:

    results = []
    progress = st.progress(0)

    for i, file in enumerate(files):
        img = Image.open(file).convert("RGB")
        img.thumbnail((800, 800))  # 軽量化

        score = calc_score(img)
        category = detect_category(img)

        results.append({
            "name": file.name,
            "img": img,
            "score": score,
            "category": category
        })

        progress.progress((i + 1) / len(files))

    # -------------------------
    # 総合ランキング
    # -------------------------
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    st.header("🏆 総合ランキング")

    for rank, item in enumerate(results, 1):
        col1, col2 = st.columns([1,2])

        with col1:
            st.image(item["img"], use_container_width=True)

        with col2:
            st.subheader(f"{rank}位 {item['name']}")
            st.metric("スコア", item["score"])
            st.write(item["category"])

        st.divider()

    # -------------------------
    # ジャンル別TOP3
    # -------------------------
    st.header("🎯 ジャンル別ランキング（TOP3）")

    categories = ["🐶 ペット", "🍜 食べ物", "🌄 風景", "😊 人物", "📷 その他"]

    for cat in categories:
        cat_items = [x for x in results if x["category"] == cat]

        if len(cat_items) == 0:
            continue

        st.subheader(cat)

        top3 = sorted(cat_items, key=lambda x: x["score"], reverse=True)[:3]

        cols = st.columns(len(top3))

        for i, item in enumerate(top3):
            with cols[i]:
                st.image(item["img"], use_container_width=True)
                st.write(f"{i+1}位")
                st.write(f"スコア: {item['score']}")

        # SNS画像（1位）
        best = top3[0]
        sns_img = create_sns_image(best["img"], f"{cat} BEST1")

        buf = io.BytesIO()
        sns_img.save(buf, format="PNG")

        st.download_button(
            f"{cat} SNS画像ダウンロード",
            buf.getvalue(),
            file_name=f"{cat}_best.png",
            mime="image/png"
        )

    # -------------------------
    # CSVダウンロード
    # -------------------------
    df = pd.DataFrame([
        {
            "順位": i+1,
            "名前": x["name"],
            "スコア": x["score"],
            "カテゴリ": x["category"]
        }
        for i, x in enumerate(results)
    ])

    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "📥 CSVダウンロード",
        csv,
        file_name="ranking.csv",
        mime="text/csv"
    )

else:
    st.info("写真をアップロードしてください")
