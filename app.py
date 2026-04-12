import streamlit as st
from PIL import Image, ImageDraw
import torch
import open_clip
import numpy as np
from pillow_heif import register_heif_opener
import io

register_heif_opener()
st.set_page_config(layout="wide")

# ===== セッション =====
if "results" not in st.session_state:
    st.session_state.results = None

# ===== UI =====
st.title("📸 AIフォトコンテスト")

selected_view = st.selectbox("表示モード", ["総合", "dog", "person", "landscape", "food"])

# ===== モデル =====
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    model.eval()
    return model, preprocess

model, preprocess = load_model()

# ===== カテゴリ =====
texts = {
    "dog":[
        "a dog",
        "cute dog",
        "pet dog"
    ],
    "person":[
        "a person",
        "people",
        "group of people",
        "family photo",
        "friends photo",
        "portrait",
        "close up face"
    ],
    "landscape":[
        "landscape",
        "nature scenery",
        "mountain",
        "outdoor view"
    ],
    "food":[
        "food",
        "delicious food",
        "meal",
        "dish",
        "restaurant food",
        "plated food"
    ]
}

# ===== テキスト特徴 =====
text_features_dict = {}
for cat, txts in texts.items():
    tokens = open_clip.tokenize(txts)
    with torch.no_grad():
        f = model.encode_text(tokens)
        f /= f.norm(dim=-1, keepdim=True)
    text_features_dict[cat] = f

# ===== 品質スコア =====
quality_texts = [
    "high quality photo",
    "well composed photo",
    "sharp photo",
    "blurry photo",
    "dark photo"
]

tokens = open_clip.tokenize(quality_texts)
with torch.no_grad():
    qf = model.encode_text(tokens)
    qf /= qf.norm(dim=-1, keepdim=True)
quality_feature = qf

# ===== maxスコア =====
def get_best_score(feat, text_features):
    sims = (feat @ text_features.T).squeeze()
    return sims.max().item()

# ===== 推論 =====
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

        cat_score = scores[best_cat] * 100
        quality_score_100 = quality_score * 100
        bright_score = max(0, 100 - abs(brightness-120))
        contrast_score = min(100, contrast * 2)

        total = (
            cat_score*0.4 +
            quality_score_100*0.3 +
            bright_score*0.2 +
            contrast_score*0.1
        )

        results.append({
            "file": file,
            "total": total,
            "cat": best_cat,
            "scores": scores,
            "detail": {
                "カテゴリ一致": round(cat_score,1),
                "品質": round(quality_score_100,1),
                "明るさ": round(bright_score,1),
                "コントラスト": round(contrast_score,1)
            }
        })

    return results

# ===== アップロード =====
uploaded_files = st.file_uploader("写真を選択", accept_multiple_files=True)

if uploaded_files:

    image_data = []
    for f in uploaded_files:
        try:
            img = Image.open(f).convert("RGB").resize((512,512))
            image_data.append((f,img))
        except:
            continue

    if st.button("ランキング実行"):
        with st.spinner("分析中..."):
            st.session_state.results = run_inference(image_data)

# ===== 表示 =====
if st.session_state.results:

    results = st.session_state.results

    if selected_view == "person":

        filtered = []

        for r in results:

            p = r["scores"]["person"]
            others = max(
                r["scores"]["dog"],
                r["scores"]["landscape"],
                r["scores"]["food"]
            )

            # 強い → 無条件OK
            if p > 0.26:
                filtered.append(r)

            # 中間 → ゆるく通す（集合写真救済）
            elif p > 0.18 and (p - others) > -0.01:
                filtered.append(r)

            # 弱い → 厳しく（レモン排除）
            elif p > 0.15 and (p - others) > 0.03:
                filtered.append(r)

        results = sorted(filtered, key=lambda r: r["scores"]["person"], reverse=True)

    elif selected_view != "総合":

        results = [
            r for r in results
            if r["cat"] == selected_view and r["scores"][selected_view] > 0.18
        ]

        results = sorted(results, key=lambda r: r["total"], reverse=True)

    else:
        results = sorted(results, key=lambda r: r["total"], reverse=True)

    st.subheader("TOP4")

    for r in results[:4]:

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(r["file"], width=250)

        with col2:
            st.markdown(f"""
### 総合スコア：{round(r['total'],1)}点  
カテゴリ：{r['cat']}

---

**内訳**
- カテゴリ一致：{r["detail"]["カテゴリ一致"]}点  
- 品質：{r["detail"]["品質"]}点  
- 明るさ：{r["detail"]["明るさ"]}点  
- コントラスト：{r["detail"]["コントラスト"]}点  

---

**カテゴリ詳細スコア**
- dog：{round(r["scores"]["dog"],2)}  
- person：{round(r["scores"]["person"],2)}  
- landscape：{round(r["scores"]["landscape"],2)}  
- food：{round(r["scores"]["food"],2)}  
""")

    # ===== インスタ画像 =====
    if st.button("インスタ画像生成"):

        W,H = 1080,1080
        canvas = Image.new("RGB",(W,H),(15,15,15))
        draw = ImageDraw.Draw(canvas)

        positions = [(0,0),(540,0),(0,540),(540,540)]

        for i,r in enumerate(results[:4]):
            img = Image.open(r["file"]).resize((540,540))
            canvas.paste(img,positions[i])
            draw.text((positions[i][0]+20,positions[i][1]+20),
                      f"#{i+1}",fill=(255,255,255))

        st.image(canvas)

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG")
        st.download_button("DL",buf.getvalue(),"insta.jpg","image/jpeg")
