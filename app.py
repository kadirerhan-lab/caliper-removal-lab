from __future__ import annotations
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.caliper import CaliperConfig, detect_caliper_mask, remove_calipers
from src.roi import ROI, crop_roi, draw_roi
from src.features import extract_features
from src.tirads import COMPOSITION, ECHOGENICITY, SHAPE, MARGIN, FOCI, calculate_tirads
from src.risk import demo_risk_score

st.set_page_config(page_title="ThyroVision MVP", page_icon="🩻", layout="wide")
st.title("🩻 ThyroVision MVP")
st.caption("Görüntü → Kaliper temizleme → Manuel ROI → Özellik analizi → TI-RADS → Demo risk skoru")
st.warning("Ar-Ge ve eğitim amaçlıdır. Klinik tanı, tedavi veya hasta yönetimi için kullanılamaz.")

if "cleaned" not in st.session_state:
    st.session_state.cleaned = None
if "mask" not in st.session_state:
    st.session_state.mask = None
if "features" not in st.session_state:
    st.session_state.features = None
if "tirads" not in st.session_state:
    st.session_state.tirads = None

tabs = st.tabs([
    "1. Görüntü",
    "2. Kaliper",
    "3. Manuel ROI",
    "4. Shape & Texture",
    "5. TI-RADS",
    "6. Risk Tahmini",
    "7. Özet",
])

with tabs[0]:
    uploaded = st.file_uploader("Ultrason görüntüsü yükle", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = np.array(Image.open(uploaded).convert("RGB"))
        st.session_state.original = image
        st.image(image, caption="Orijinal görüntü", use_container_width=True)
        st.success("Görüntü yüklendi. Kaliper sekmesine geçebilirsin.")
    else:
        st.info("Başlamak için bir PNG/JPG görüntüsü yükle.")

with tabs[1]:
    if "original" not in st.session_state:
        st.info("Önce 1. Görüntü sekmesinden dosya yükle.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            threshold = st.slider("Parlaklık eşiği", 150, 255, 220)
            dilation = st.slider("Maske genişletme", 0, 8, 2)
        with c2:
            radius = st.slider("Inpainting yarıçapı", 1, 15, 3)
            method = st.selectbox("Yöntem", ["TELEA", "NAVIER_STOKES"])
        if st.button("Kaliperi tespit et ve kaldır", type="primary"):
            cfg = CaliperConfig(brightness_threshold=threshold, dilation=dilation,
                                inpaint_radius=radius, method=method)
            mask = detect_caliper_mask(st.session_state.original, cfg)
            cleaned = remove_calipers(st.session_state.original, mask, cfg)
            st.session_state.mask = mask
            st.session_state.cleaned = cleaned
        if st.session_state.cleaned is not None:
            a, b, c = st.columns(3)
            a.image(st.session_state.original, caption="Orijinal", use_container_width=True)
            b.image(st.session_state.mask, caption="Kaliper maskesi", use_container_width=True)
            c.image(st.session_state.cleaned, caption="Temizlenmiş", use_container_width=True)

with tabs[2]:
    base = st.session_state.cleaned if st.session_state.cleaned is not None else st.session_state.get("original")
    if base is None:
        st.info("Önce görüntü yükle.")
    else:
        h, w = base.shape[:2]
        st.write("Nodülü çevreleyen dikdörtgen alanı kaydırıcılarla belirle.")
        x = st.slider("X", 0, max(0, w-1), int(w*0.25))
        y = st.slider("Y", 0, max(0, h-1), int(h*0.25))
        rw = st.slider("Genişlik", 10, max(10, w-x), max(10, min(int(w*0.4), w-x)))
        rh = st.slider("Yükseklik", 10, max(10, h-y), max(10, min(int(h*0.4), h-y)))
        roi = ROI(x, y, rw, rh)
        st.session_state.roi = roi
        left, right = st.columns(2)
        left.image(draw_roi(base, roi), caption="ROI konumu", use_container_width=True)
        right.image(crop_roi(base, roi), caption="Nodül ROI", use_container_width=True)

with tabs[3]:
    base = st.session_state.cleaned if st.session_state.cleaned is not None else st.session_state.get("original")
    roi = st.session_state.get("roi")
    if base is None or roi is None:
        st.info("Önce görüntü yükle ve manuel ROI belirle.")
    else:
        threshold = st.slider("Basit segmentasyon eşiği", 20, 220, 90)
        roi_img = crop_roi(base, roi)
        features, nodule_mask = extract_features(roi_img, threshold)
        st.session_state.features = features
        c1, c2 = st.columns(2)
        c1.image(roi_img, caption="ROI", use_container_width=True)
        c2.image(nodule_mask, caption="Tahmini nodül maskesi", use_container_width=True)
        df = pd.DataFrame([features]).T.reset_index()
        df.columns = ["Özellik", "Değer"]
        st.dataframe(df, use_container_width=True, hide_index=True)

with tabs[4]:
    composition = st.selectbox("Composition", list(COMPOSITION))
    echogenicity = st.selectbox("Echogenicity", list(ECHOGENICITY))
    shape = st.selectbox("Shape", list(SHAPE))
    margin = st.selectbox("Margin", list(MARGIN))
    foci = st.multiselect("Echogenic foci", list(FOCI), default=["Yok veya büyük kuyruklu artefakt"])
    if st.button("TI-RADS puanını hesapla", type="primary"):
        st.session_state.tirads = calculate_tirads(composition, echogenicity, shape, margin, foci)
    if st.session_state.tirads:
        st.metric("Kategori", st.session_state.tirads["category"])
        st.metric("Toplam puan", st.session_state.tirads["score"])
        st.caption("Bu ekran kullanıcı girdisine dayalı TI-RADS simülasyonudur.")

with tabs[5]:
    if st.session_state.features is None or st.session_state.tirads is None:
        st.info("Önce özellik analizi ve TI-RADS adımlarını tamamla.")
    else:
        risk = demo_risk_score(st.session_state.tirads["score"], st.session_state.features)
        st.session_state.risk = risk
        a, b = st.columns(2)
        a.metric("Benign demo olasılığı", f'%{risk["benign_probability_demo"]:.1f}')
        b.metric("Malignant demo olasılığı", f'%{risk["malignant_probability_demo"]:.1f}')
        st.error("Bu değer eğitilmiş klinik model sonucu değildir; yalnızca MVP akışını göstermek için üretilen kural tabanlı demo skorudur.")

with tabs[6]:
    st.subheader("MVP Analiz Özeti")
    summary = {
        "Görüntü yüklendi": "original" in st.session_state,
        "Kaliper işlemi": st.session_state.cleaned is not None,
        "ROI belirlendi": st.session_state.get("roi") is not None,
        "Özellikler hesaplandı": st.session_state.features is not None,
        "TI-RADS hesaplandı": st.session_state.tirads is not None,
        "Demo risk skoru": st.session_state.get("risk") is not None,
    }
    st.dataframe(pd.DataFrame(summary.items(), columns=["Adım", "Tamamlandı"]), hide_index=True, use_container_width=True)
    if st.session_state.tirads:
        st.write(f'**TI-RADS:** {st.session_state.tirads["category"]} ({st.session_state.tirads["score"]} puan)')
    if st.session_state.get("risk"):
        st.write(f'**Demo malignant olasılığı:** %{st.session_state.risk["malignant_probability_demo"]:.1f}')
    st.caption("Sonraki aşamada bu özet JSON/PDF rapor olarak dışa aktarılabilir.")
