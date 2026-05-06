# app.py

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import io

st.set_page_config(page_title="写真ランキングAI", layout="wide")

st.title("📷 写真ランキングAI（強化版）")
st.caption("ジャンル別ランキング＋SNSシェア画像生成")

# -------------------------
# スコア計算（改良版）
# -------------------------
def calc_score(img):
    arr = np.array(img)

    brightness = arr.mean()
    contrast = arr.std()
    colorfulness = np.std(arr[:,:,0]) + np.std(arr[:,:,1]) + np.std(arr[:,:,2])

    # 少しランダム要素追加（重要）
    randomness = np.random.uniform(0, 15)

    score = brightness * 0.2 + contrast * 0.3 + colorfulness * 0.3 + randomness
    return round(score, 1)

# -------------------------
# カテゴリ判定
# -------------------------
def detect_category(filename):
    name = filename.lower()

    if any(x in name for x in ["dog", "inu", "pet"]):
        return "🐶 ペット"
    elif any(x in name for x in ["food", "meal", "lunch"]):
        return "🍜 食べ物"
    elif any(x in name for x in ["sea", "mountain", "sky"]):
        return "🌄 風景"
    elif any(x in name for x in ["face", "selfie", "person"]):
        return "😊 人物"
    else:
        return "📷 その他"

# -------------------------
# コメント（ジャンル別）
# -------------------------
def make_comment(category, score):
    if category == "🐶 ペット":
        return "表情が最高！癒し力高めです。"
    elif category == "🍜 食べ物":
        return "シズル感があって美味しそう！"
    elif category == "🌄 風景":
        return "構図が良くて引き込まれます。"
    elif category == "😊 人物":
        return "自然な表情が魅力的です。"
    else:
        return "バランスの良い一枚です。"

# -------------------------
# SNS画像生成
# -------------------------
def create_sns_image(img, title):
    base = img.copy().resize((600, 600))

    canvas = Image.new("RGB", (600, 700), "white")
    canvas.paste(base, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # フォント（なければデフォルト）
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    draw.text((20, 620), title, fill="black", font=font)

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
        img.thumbnail((800, 800))

        score = calc_score(img)
        category = detect_category(file.name)
        comment = make_comment(category, score)

        results.append({
            "name": file.name,
            "img": img,
            "score": score,
            "category": category,
            "comment": comment
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
            st.write(item["comment"])

        st.divider()

    # -------------------------
    # ジャンル別TOP3
    # -------------------------
    st.header("🎯 ジャンル別ベスト3")

    categories = list(set([x["category"] for x in results]))

    for cat in categories:
        st.subheader(cat)

        cat_items = [x for x in results if x["category"] == cat]
        cat_items = sorted(cat_items, key=lambda x: x["score"], reverse=True)[:3]

        cols = st.columns(3)

        for i, item in enumerate(cat_items):
            with cols[i]:
                st.image(item["img"], use_container_width=True)
                st.write(f"{i+1}位")
                st.write(f"スコア: {item['score']}")

        # SNS画像生成
        if len(cat_items) > 0:
            sns_img = create_sns_image(cat_items[0]["img"], f"{cat} BEST1")

            buf = io.BytesIO()
            sns_img.save(buf, format="PNG")

            st.download_button(
                f"{cat} SNS画像ダウンロード",
                buf.getvalue(),
                file_name=f"{cat}_best.png",
                mime="image/png"
            )

    # -------------------------
    # CSV保存
    # -------------------------
    df = pd.DataFrame([
        {
            "順位": i+1,
            "名前": x["name"],
            "スコア": x["score"],
            "カテゴリ": x["category"],
            "コメント": x["comment"]
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
