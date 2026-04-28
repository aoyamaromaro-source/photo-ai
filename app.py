# app.py
# 写真ランキングアプリ（軽量版フルコード）
# Streamlit Cloud対応 / GPTなし / 軽量高速版

import streamlit as st
from PIL import Image
import os
import io
import zipfile
import tempfile
import numpy as np

st.set_page_config(page_title="写真ランキングAI", layout="wide")

st.title("📷 写真ランキングアプリ（軽量版）")
st.caption("アップロードした写真をAI風に自動採点・ランキングします")

# -------------------------
# スコア計算（超軽量）
# -------------------------
def calc_score(img):
    arr = np.array(img)

    # 明るさ
    brightness = arr.mean()

    # コントラスト
    contrast = arr.std()

    # 色の豊かさ
    colorfulness = np.std(arr[:,:,0]) + np.std(arr[:,:,1]) + np.std(arr[:,:,2])

    # 総合点
    score = brightness * 0.3 + contrast * 0.4 + colorfulness * 0.3
    return round(score, 1)

# -------------------------
# カテゴリ判定（簡易）
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
# コメント生成
# -------------------------
def make_comment(score):
    if score >= 120:
        return "かなり魅力的な一枚。SNS映えレベルです。"
    elif score >= 100:
        return "バランスの良い写真です。"
    elif score >= 80:
        return "自然で見やすい写真です。"
    else:
        return "少し暗いか単調かもしれません。"

# -------------------------
# アップロード
# -------------------------
files = st.file_uploader(
    "写真を複数選択してください",
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
        comment = make_comment(score)

        results.append({
            "name": file.name,
            "img": img,
            "score": score,
            "category": category,
            "comment": comment
        })

        progress.progress((i + 1) / len(files))

    # ランキング
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    st.success("ランキング完成！")

    # 表示
    for rank, item in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([1,2])

            with col1:
                st.image(item["img"], use_container_width=True)

            with col2:
                st.subheader(f"{rank}位　{item['name']}")
                st.metric("スコア", item["score"])
                st.write(item["category"])
                st.write(item["comment"])

            st.divider()

    # -------------------------
    # CSV保存
    # -------------------------
    import pandas as pd

    df = pd.DataFrame([
        {
            "順位": i+1,
            "ファイル名": x["name"],
            "スコア": x["score"],
            "カテゴリ": x["category"],
            "コメント": x["comment"]
        }
        for i, x in enumerate(results)
    ])

    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "📥 ランキングCSVダウンロード",
        csv,
        file_name="photo_ranking.csv",
        mime="text/csv"
    )

else:
    st.info("写真をアップロードするとランキング開始します")
