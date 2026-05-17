import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ── Load model (sekali saja, di-cache) ────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import joblib
        model    = joblib.load("predictive_maintenance_model.pkl")
        features = joblib.load("predictive_maintenance_features.pkl")
        return model, features
    except Exception as e:
        return None, None

model, feature_names = load_model()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
)

# ── CSS minimal (hanya animasi & button yg tidak bisa inline) ─────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0f1117 !important;
    color: #e0e0e0 !important;
    font-family: 'Segoe UI', sans-serif;
}
.block-container { padding: 1.2rem 1.5rem 2rem 1.5rem !important; }
section[data-testid="stSidebar"] { background-color: #1a1d27 !important; }
.stButton > button {
    background: linear-gradient(135deg,#4f6ef7,#7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; width: 100% !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 10px 0 !important;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Input Sensor")
    st.markdown("---")
    suhu_udara   = st.slider("🌡️ Suhu Udara (K)",          270,  320, 300)
    suhu_proses  = st.slider("🔥 Suhu Proses (K)",          300,  340, 310)
    kecepatan    = st.slider("⚡ Kecepatan Putaran (RPM)", 1000, 3000, 1202)
    torsi        = st.slider("🔩 Torsi (Nm)",               10,   100,  70)
    keausan_alat = st.slider("🔧 Keausan Alat (min)",         0,   300, 220)
    st.markdown("---")
    st.button("🔍 Prediksi Sekarang", use_container_width=True)

    # Tampilkan info sumber prediksi di sidebar
    if model is not None:
        st.markdown(
            '<div style="font-size:11px;color:#4ade80;text-align:center;margin-top:8px;">'
            '✅ Model ML aktif</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:11px;color:#facc15;text-align:center;margin-top:8px;">'
            '⚠️ Menggunakan rule-based (model .pkl tidak ditemukan)</div>',
            unsafe_allow_html=True,
        )

# ── Helper functions ───────────────────────────────────────────────────────────
def predict(su, sp, kec, tor, kau):
    """
    Prediksi menggunakan model ML jika tersedia.
    Fallback ke rule-based jika model tidak bisa diload.
    """
    if model is not None:
        # ── ML model path ──────────────────────────────────────────────────────
        # Urutan fitur harus sama persis dengan saat training
        input_data = np.array([[su, sp, kec, tor, kau]])

        # predict_proba() → [[prob_class_0, prob_class_1]]
        proba = model.predict_proba(input_data)[0]
        prob  = int(round(proba[1] * 100))   # probabilitas kegagalan (class 1)

        # Deteksi tipe kegagalan dari feature importance model
        if kau > 180 and tor > 60:
            ft = "Tool Wear Failure"
        elif sp > 325:
            ft = "Heat Dissipation Failure"
        elif kec < 1100:
            ft = "Power Failure"
        else:
            ft = "Random Failure"

        return prob, ft

    else:
        # ── Rule-based fallback ────────────────────────────────────────────────
        score = 0
        if kau > 200:   score += 40
        elif kau > 150: score += 20
        if tor > 60:    score += 25
        elif tor > 45:  score += 12
        if sp > 320:    score += 15
        if kec < 1300:  score += 10
        if su > 305:    score += 10
        prob = min(score, 99)
        if kau > 180 and tor > 60: ft = "Tool Wear Failure"
        elif sp > 325:             ft = "Heat Dissipation Failure"
        elif kec < 1100:           ft = "Power Failure"
        else:                      ft = "Random Failure"
        return prob, ft


def get_feature_importance():
    """
    Ambil feature importance dari model jika tersedia.
    Fallback ke nilai hardcode jika model tidak ada.
    Returns: list of (nama, pct) sorted descending
    """
    label_map = {
        "suhu_udara":   "Suhu Udara",
        "suhu_proses":  "Suhu Proses",
        "kecepatan":    "RPM",
        "torsi":        "Torsi",
        "keausan_alat": "Keausan Alat",
        # Nama alternatif yang mungkin dipakai saat training (AI4I dataset)
        "Air temperature [K]":          "Suhu Udara",
        "Process temperature [K]":      "Suhu Proses",
        "Rotational speed [rpm]":       "RPM",
        "Torque [Nm]":                  "Torsi",
        "Tool wear [min]":              "Keausan Alat",
    }

    if model is not None and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        # Gunakan feature_names dari .pkl jika ada, fallback ke urutan default
        names = feature_names if feature_names is not None else [
            "suhu_udara", "suhu_proses", "kecepatan", "torsi", "keausan_alat"
        ]

        # Normalisasi ke 0–100
        max_imp = max(importances) if max(importances) > 0 else 1
        pairs   = []
        for fname, imp in zip(names, importances):
            display = label_map.get(fname, fname)
            pct     = int(round(imp / max_imp * 100))
            pairs.append((display, pct))

        # Sort descending
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:5]   # top 5

    # Hardcode fallback
    return [
        ("Keausan Alat", 88),
        ("Torsi",        62),
        ("Suhu Proses",  45),
        ("RPM",          28),
        ("Suhu Udara",   14),
    ]


def get_level(p):
    if p >= 70: return "Kritis",    "kritis"
    if p >= 40: return "Peringatan","warning"
    return "Normal", "normal"

def sensor_status(val, lo, hi):
    if val > hi: return "Melebihi batas",  "danger"
    if val < lo: return "Di bawah normal", "warning"
    return "Normal", "normal"

def biz_info(p):
    if p >= 70: return "Rp 42.000.000", "6–8 jam", "Ganti tool sekarang"
    if p >= 40: return "Rp 18.000.000", "2–4 jam", "Jadwalkan maintenance"
    return "Rp 0", "—", "Tidak diperlukan"

# ── Hitung nilai ───────────────────────────────────────────────────────────────
prob, failure_type        = predict(suhu_udara, suhu_proses, kecepatan, torsi, keausan_alat)
label, level              = get_level(prob)
cost, durasi, rekomendasi = biz_info(prob)
fi_data                   = get_feature_importance()

STATUS_CLR = {"normal": "#4ade80", "warning": "#facc15", "danger": "#f87171"}

BADGE = {
    "kritis":  ("rgba(220,38,38,0.2)",  "rgba(220,38,38,0.6)",  "#f87171"),
    "warning": ("rgba(234,179,8,0.15)", "rgba(234,179,8,0.5)",  "#facc15"),
    "normal":  ("rgba(34,197,94,0.15)", "rgba(34,197,94,0.5)",  "#4ade80"),
}
ALERT = {
    "kritis":  ("rgba(220,38,38,0.15)",  "rgba(220,38,38,0.5)",  "#f87171",
                "⚠️", "Kegagalan diprediksi dalam waktu dekat",
                "Keausan alat dan torsi melebihi ambang batas aman — segera lakukan inspeksi"),
    "warning": ("rgba(234,179,8,0.13)",  "rgba(234,179,8,0.45)", "#facc15",
                "⚡", "Perhatian — parameter mendekati batas",
                "Beberapa sensor menunjukkan nilai di atas normal, pantau secara berkala"),
    "normal":  ("rgba(34,197,94,0.12)",  "rgba(34,197,94,0.4)",  "#4ade80",
                "✅", "Semua parameter dalam kondisi normal",
                "Mesin beroperasi dalam rentang aman — tidak ada tindakan diperlukan"),
}

abg, aborder, aclr, aicon, atitle, asub = ALERT[level]
bbg, bborder, bclr                      = BADGE[level]
gauge_clr = "#ef4444" if prob >= 70 else "#f59e0b" if prob >= 40 else "#22c55e"
cost_clr  = "#f87171" if prob >= 70 else "#fb923c" if prob >= 40 else "#4ade80"
rek_clr   = "#fb923c" if prob >= 70 else "#facc15" if prob >= 40 else "#4ade80"

# ── HEADER ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
        f'  <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:8px;'
        f'              width:36px;height:36px;display:flex;align-items:center;'
        f'              justify-content:center;font-size:18px;">⚙️</div>'
        f'  <div>'
        f'    <div style="font-size:17px;font-weight:700;color:#f0f0f0;">Mesin CNC — Line 3</div>'
        f'    <div style="font-size:12px;color:#6b7280;">PT Industri Maju · Semarang Plant</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:flex-end;gap:10px;margin-top:6px;">'
        f'  <span style="font-size:12px;color:#9ca3af;">'
        f'    <span style="display:inline-block;width:8px;height:8px;background:#4ade80;'
        f'                 border-radius:50%;margin-right:5px;'
        f'                 animation:blink 1.2s infinite;"></span>Live monitoring'
        f'  </span>'
        f'  <span style="background:{bbg};border:1px solid {bborder};color:{bclr};'
        f'               border-radius:6px;padding:3px 10px;font-size:12px;font-weight:600;">'
        f'    ⚠️ Risiko {label}'
        f'  </span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── ALERT BANNER ───────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:{abg};border:1px solid {aborder};border-radius:8px;'
    f'            padding:14px 18px;margin-bottom:16px;display:flex;align-items:flex-start;gap:12px;">'
    f'  <span style="font-size:20px;margin-top:1px;">{aicon}</span>'
    f'  <div>'
    f'    <div style="color:{aclr};font-weight:700;font-size:14px;margin-bottom:3px;">{atitle}</div>'
    f'    <div style="color:#9ca3af;font-size:12px;">{asub}</div>'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── ROW 1: GAUGE | BISNIS + FEATURE IMPORTANCE ────────────────────────────────
col1, col2 = st.columns([1, 1])

# ── GAUGE ──────────────────────────────────────────────────────────────────────
with col1:
    st.markdown(
        '<div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;'
        'padding:16px 20px;margin-bottom:14px;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:11px;font-weight:600;letter-spacing:1.2px;color:#6b7280;'
        'text-transform:uppercase;margin-bottom:14px;">Probabilitas Kegagalan</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%", "font": {"size": 36, "color": gauge_clr}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#0f1117",
                     "tickvals": [], "visible": False},
            "bar": {"color": gauge_clr, "thickness": 0.28},
            "bgcolor": "#12151f",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   40], "color": "#12151f"},
                {"range": [40,  70], "color": "#12151f"},
                {"range": [70, 100], "color": "#12151f"},
            ],
            "threshold": {"line": {"color": gauge_clr, "width": 3},
                          "thickness": 0.8, "value": prob},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=200, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
    )
    fig.add_annotation(x=0.12, y=0.12, text="Aman",              showarrow=False,
                       font=dict(size=10, color="#6b7280"), xref="paper", yref="paper")
    fig.add_annotation(x=0.88, y=0.12, text="Kritis",            showarrow=False,
                       font=dict(size=10, color="#6b7280"), xref="paper", yref="paper")
    fig.add_annotation(x=0.50, y=0.02, text="probabilitas gagal", showarrow=False,
                       font=dict(size=10, color="#6b7280"), xref="paper", yref="paper")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f'<div style="background:#12151f;border:1px solid #2a2d3a;border-radius:8px;'
        f'            padding:10px 14px;display:flex;align-items:center;gap:10px;margin-top:6px;">'
        f'  <span style="font-size:18px;">🔧</span>'
        f'  <div>'
        f'    <div style="font-size:11px;color:#6b7280;">Tipe kegagalan terdeteksi</div>'
        f'    <div style="font-size:13px;color:#e0e0e0;font-weight:600;">{failure_type}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── BISNIS + FEATURE IMPORTANCE ───────────────────────────────────────────────
with col2:

    st.markdown(
        '<div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;'
        'padding:16px 20px;margin-bottom:14px;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:11px;font-weight:600;letter-spacing:1.2px;color:#6b7280;'
        'text-transform:uppercase;margin-bottom:14px;">Estimasi Dampak Bisnis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'            padding:8px 0;border-bottom:1px solid #2a2d3a;font-size:13px;">'
        f'  <span style="color:#9ca3af;">Downtime cost</span>'
        f'  <span style="color:{cost_clr};font-weight:600;">{cost}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'            padding:8px 0;border-bottom:1px solid #2a2d3a;font-size:13px;">'
        f'  <span style="color:#9ca3af;">Est. durasi downtime</span>'
        f'  <span style="color:#e0e0e0;font-weight:600;">{durasi}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'            padding:8px 0;font-size:13px;">'
        f'  <span style="color:#9ca3af;">Rekomendasi tindakan</span>'
        f'  <span style="color:{rek_clr};font-weight:600;">{rekomendasi}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;'
        'padding:16px 20px;margin-bottom:14px;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:11px;font-weight:600;letter-spacing:1.2px;color:#6b7280;'
        'text-transform:uppercase;margin-bottom:14px;">Feature Importance</div>',
        unsafe_allow_html=True,
    )

    # fi_data berasal dari model asli atau fallback hardcode
    for fname, fpct in fi_data:
        fa, fb = st.columns([4, 1])
        with fa:
            st.markdown(
                f'<div style="font-size:12px;color:#9ca3af;margin-bottom:3px;">{fname}</div>'
                f'<div style="background:#2a2d3a;border-radius:4px;height:6px;margin-bottom:8px;">'
                f'  <div style="height:6px;border-radius:4px;background:#4f6ef7;width:{fpct}%;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with fb:
            st.markdown(
                f'<div style="font-size:12px;color:#e0e0e0;font-weight:600;'
                f'text-align:right;padding-top:1px;">{fpct}%</div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ── ROW 2: STATUS SENSOR ──────────────────────────────────────────────────────
sensors = [
    {"icon": "🌡️", "name": "Suhu Udara",       "value": f"{suhu_udara} K",
     "pct": (suhu_udara - 270) / 50,
     "status": sensor_status(suhu_udara, 275, 308),
     "icon_bg": "rgba(34,197,94,0.15)"},
    {"icon": "🔥", "name": "Suhu Proses",       "value": f"{suhu_proses} K",
     "pct": (suhu_proses - 300) / 40,
     "status": sensor_status(suhu_proses, 305, 325),
     "icon_bg": "rgba(34,197,94,0.15)"},
    {"icon": "⚡", "name": "Kecepatan Putaran", "value": f"{kecepatan} RPM",
     "pct": (kecepatan - 1000) / 2000,
     "status": sensor_status(kecepatan, 1300, 2800),
     "icon_bg": "rgba(234,179,8,0.15)"},
    {"icon": "🔩", "name": "Torsi",             "value": f"{torsi} Nm",
     "pct": torsi / 100,
     "status": sensor_status(torsi, 10, 65),
     "icon_bg": "rgba(220,38,38,0.15)"},
    {"icon": "🔧", "name": "Keausan Alat",      "value": f"{keausan_alat} min",
     "pct": keausan_alat / 300,
     "status": sensor_status(keausan_alat, 0, 200),
     "icon_bg": "rgba(220,38,38,0.15)"},
]

st.markdown(
    '<div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;'
    'padding:16px 20px;margin-bottom:14px;">',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
    '  <div style="font-size:11px;font-weight:600;letter-spacing:1.2px;color:#6b7280;'
    '              text-transform:uppercase;">Status Sensor</div>'
    '  <span style="font-size:11px;color:#6b7280;">5 parameter aktif</span>'
    '</div>',
    unsafe_allow_html=True,
)

for idx, s in enumerate(sensors):
    slabel, stype = s["status"]
    sc  = STATUS_CLR[stype]
    pct = min(max(s["pct"], 0.0), 1.0) * 100

    c_icon, c_bar, c_val = st.columns([2, 5, 2])

    with c_icon:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;">'
            f'  <div style="background:{s["icon_bg"]};width:32px;height:32px;border-radius:8px;'
            f'              display:flex;align-items:center;justify-content:center;'
            f'              font-size:15px;flex-shrink:0;">{s["icon"]}</div>'
            f'  <span style="font-size:13px;color:#d1d5db;font-weight:500;">{s["name"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c_bar:
        st.markdown(
            f'<div style="margin-top:15px;position:relative;">'
            f'  <div style="background:#2a2d3a;border-radius:4px;height:6px;width:100%;">'
            f'    <div style="height:6px;border-radius:4px;background:{sc};width:{pct:.1f}%;"></div>'
            f'  </div>'
            f'  <div style="position:absolute;top:0;left:70%;width:1px;height:6px;'
            f'              background:rgba(255,255,255,0.3);"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c_val:
        st.markdown(
            f'<div style="text-align:right;padding:4px 0;">'
            f'  <div style="font-size:14px;font-weight:700;color:#e0e0e0;">{s["value"]}</div>'
            f'  <div style="font-size:11px;color:{sc};">{slabel}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if idx < len(sensors) - 1:
        st.markdown(
            '<hr style="border:none;border-top:1px solid #1f2230;margin:2px 0;">',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#374151;font-size:11px;margin-top:20px;">'
    'Predictive Maintenance AI · Semarang Plant · Real-time monitoring'
    '</div>',
    unsafe_allow_html=True,
)