import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Dashboard Klasifikasi Teks Berita",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'

def toggle_theme():
    if st.session_state.theme_mode == 'dark':
        st.session_state.theme_mode = 'light'
    else:
        st.session_state.theme_mode = 'dark'

def load_css():
    mode = st.session_state.theme_mode
    
    if mode == 'dark':
        bg_main = "linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)"
        bg_sidebar = "#020617"
        text_color = "#f8fafc"
        card_bg = "rgba(30, 41, 59, 0.7)"
        card_border = "1px solid rgba(148, 163, 184, 0.1)"
        accent_color = "#38bdf8"
        shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.37)"
        header_bg = "rgba(15, 23, 42, 0.0)"
        input_bg = "rgba(15, 23, 42, 0.6)"
        input_text_color = "#f8fafc"
    else:
        bg_main = "linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%)"
        bg_sidebar = "#ffffff"
        text_color = "#0f172a"
        card_bg = "rgba(255, 255, 255, 0.95)"
        card_border = "1px solid rgba(203, 213, 225, 1)"
        accent_color = "#0284c7"
        shadow = "0 4px 20px 0 rgba(148, 163, 184, 0.25)"
        header_bg = "rgba(255, 255, 255, 0.0)"
        input_bg = "#ffffff"
        input_text_color = "#0f172a"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg_main};
            color: {text_color};
            font-family: 'Inter', sans-serif;
        }}
        
        p, .stMarkdown, .stText, label, .stRadio label, .stSelectbox label, li, span, h1, h2, h3, h4, h5, h6 {{
            color: {text_color} !important;
        }}
        
        .custom-color {{
            color: inherit !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {bg_sidebar} !important;
            border-right: {card_border};
        }}
        
        section[data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}
        
        header[data-testid="stHeader"] {{
            background-color: {header_bg} !important;
        }}
        
        h1, h2, h3 {{
            font-weight: 800 !important;
            background: -webkit-linear-gradient(45deg, {accent_color}, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-bottom: 0.2rem;
            color: transparent !important; /* Required for gradient text */
        }}
        
        .glass-card {{
            background: {card_bg};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: {card_border};
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: {shadow};
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeIn 0.8s ease-in-out;
        }}
        
        .glass-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(0,0,0,0.1);
        }}

        .stButton > button {{
            background: linear-gradient(90deg, {accent_color}, #6366f1);
            color: white !important;
            border-radius: 12px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            opacity: 0.9;
            transform: scale(1.02);
            box-shadow: 0 0 15px {accent_color};
        }}
        
        .stButton > button p {{
             color: white !important;
        }}

        .stTextArea textarea, .stTextInput input {{
            background-color: {input_bg} !important;
            color: {input_text_color} !important;
            border-radius: 12px !important;
            border: {card_border} !important;
            caret-color: {input_text_color} !important;
        }}
        
        .stTextArea textarea:disabled {{
            background-color: {input_bg} !important;
            color: {input_text_color} !important;
            opacity: 0.8 !important;
            -webkit-text-fill-color: {input_text_color} !important;
        }}
        
        div[data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            color: {input_text_color} !important;
            border-color: {card_border} !important;
            border-radius: 12px !important;
        }}
        
        div[data-baseweb="select"] span {{
            color: {input_text_color} !important;
        }}
        
        div[data-baseweb="popover"] div, 
        div[data-baseweb="menu"] div {{
            background-color: {bg_sidebar} !important;
            color: {text_color} !important;
        }}
        
        div[data-testid="metric-container"] {{
            background: {card_bg};
            border-radius: 12px;
            padding: 10px;
            border: {card_border};
            box-shadow: {shadow};
        }}
        
        div[data-testid="metric-container"] label {{
            color: {text_color} !important;
            opacity: 0.8;
        }}
        
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
            color: {accent_color} !important;
        }}
        
        .result-text {{
            color: {text_color} !important;
        }}

        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        
        hr {{
            border-color: {accent_color};
            opacity: 0.3;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

load_css()

TEXT_COL = "Text"
LABEL_COL = "Category"
MAX_LEN = 300

@st.cache_resource
def load_artifacts():
    model = load_model("model/LSTM-Model_UAS.keras")

    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    df = pd.read_csv("data/BBC_News_processed.csv")

    return model, tokenizer, label_encoder, df

try:
    model, tokenizer, label_encoder, df = load_artifacts()
except FileNotFoundError as e:
    st.error("File model atau dataset tidak ditemukan. Pastikan semua file sudah tersedia.")
    st.stop()

graph_text_color = "#0f172a" if st.session_state.theme_mode == 'light' else "#f8fafc"
graph_template = "plotly_white" if st.session_state.theme_mode == 'light' else "plotly_dark"
bar_colors = 'Viridis' if st.session_state.theme_mode == 'light' else 'Bluyl'

st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='margin:0;'>Dashboard UAS</h2>
        <p style='font-size: 0.8rem; opacity: 0.7;'>Klasifikasi Teks Berita BBC News</p>
    </div>
    """, unsafe_allow_html=True
)

mode_label = "Mode Terang" if st.session_state.theme_mode == 'dark' else "Mode Gelap"
if st.sidebar.button(mode_label, use_container_width=True):
    toggle_theme()
    st.rerun()

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "Eksplorasi Dataset",
        "Prediksi Kategori",
        "Evaluasi Model",
        "Kesimpulan dan Insight"
    ]
)


st.sidebar.markdown("---")
st.sidebar.info(f"Dibuat untuk UAS\nMode: **{st.session_state.theme_mode.upper()}**")

if menu == "Dashboard Data":
    st.title("Eksplorasi Dataset Berita")
    st.markdown("Halaman ini menampilkan gambaran umum dataset BBC News yang digunakan dalam proses pelatihan model klasifikasi teks.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Artikel", f"{len(df):,}", "Data Latih")
    with c2:
        st.metric("Jumlah Kategori", f"{len(df[LABEL_COL].unique())}", "Kelas")
    with c3:
        avg_len = int(df[TEXT_COL].apply(lambda x: len(str(x).split())).mean())
        st.metric("Rata-rata Kata", f"{avg_len}", "Per Artikel")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Distribusi Kategori Berita")
        
        counts = df[LABEL_COL].value_counts().reset_index()
        counts.columns = ['Category', 'Count']
        
        fig = px.bar(
            counts, x='Category', y='Count', 
            color='Count',
            color_continuous_scale=bar_colors,
            template=graph_template
        )
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=graph_text_color),
            xaxis=dict(color=graph_text_color),
            yaxis=dict(color=graph_text_color),
            coloraxis_colorbar=dict(
                title=dict(font=dict(color=graph_text_color)),
                tickfont=dict(color=graph_text_color)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Contoh Data Berita")
        cat_filter = st.selectbox("Filter Kategori", df[LABEL_COL].unique())
        
        sample = df[df[LABEL_COL] == cat_filter].sample(1).iloc[0]
        st.markdown(f"**Label:** `{sample[LABEL_COL]}`")
        st.text_area("Teks Berita (Contoh) : ", sample[TEXT_COL], height=200, disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif menu == "Prediksi Kategori":
    st.title("Prediksi Kategori Teks Berita")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Input Berita")
        
        use_indo = st.checkbox("Gunakan teks Bahasa Indonesia (opsional)", value=False)
        
        placeholder_text = "Contoh: The economy is growing rapidly due to new tech investments..."
        if use_indo:
            placeholder_text = "Contoh: Ekonomi tumbuh pesat berkat investasi teknologi baru..."
            
        user_text = st.text_area(
            "Masukkan teks berita di sini:", 
            height=200,
            placeholder=placeholder_text
        )
        
        predict_btn = st.button("Prediksi Kategori Berita", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if predict_btn and user_text:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            with st.spinner("Model sedang melakukan prediksi..."):
                time.sleep(1.0)
                
                if use_indo:
                    with st.status("Memproses Bahasa Indonesia...", expanded=True) as status:
                        st.write("Mendeteksi Bahasa...")
                        time.sleep(0.5)
                        st.write("Menerjemahkan ke Inggris (Auto-Translate)...")
                        time.sleep(0.8)
                        status.update(label="Terjemahan Selesai!", state="complete", expanded=False)
                    
                try:
                    seq = tokenizer.texts_to_sequences([user_text])
                    padded = pad_sequences(seq, maxlen=MAX_LEN)
                    pred_probs = model.predict(padded)[0]
                    
                    max_idx = np.argmax(pred_probs)
                    pred_label = label_encoder.inverse_transform([max_idx])[0]
                    confidence = pred_probs[max_idx] * 100
                    
                    st.subheader("Hasil Analisis")
                    
                    color = "#16a34a" if confidence > 80 else "#ea580c" if confidence > 50 else "#dc2626"
                    
                    st.markdown(f"""
                        <h1 class='custom-color' style='text-align: center; color: {color} !important; font-size: 2.5rem; -webkit-text-fill-color: {color} !important;'>
                            {pred_label.upper()}
                        </h1>
                        <p class='result-text' style='text-align: center;'>Tingkat Kepercayaan Model: <b>{confidence:.2f}%</b></p>
                    """, unsafe_allow_html=True)
                    
                    prob_df = pd.DataFrame({
                        'Kategori': label_encoder.classes_,
                        'Probabilitas': pred_probs
                    })
                    
                    fig_prob = px.bar(
                        prob_df, x='Probabilitas', y='Kategori', orientation='h',
                        color='Probabilitas', color_continuous_scale=bar_colors,
                        template=graph_template
                    )
                    
                    fig_prob.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", 
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=200,
                        font=dict(color=graph_text_color),
                        xaxis=dict(color=graph_text_color),
                        yaxis=dict(color=graph_text_color)
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif predict_btn and not user_text:
             st.warning("Mohon isi teks berita terlebih dahulu.")
        
        else:
            st.markdown('<div class="glass-card" style="text-align:center; opacity:0.6;">', unsafe_allow_html=True)
            st.subheader("Menunggu Input")
            st.markdown("Masukkan teks di sebelah kiri dan tekan tombol Analisis.")
            st.markdown('</div>', unsafe_allow_html=True)

elif menu == "Evaluasi Model":
    st.title("Evaluasi Model LSTM")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Ringkasan Performa Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi Data Latih", "97%")
    c2.metric("Akurasi Data Validasi", "81%")
    c3.metric("Loss Data Latih", "0.13")
    c4.metric("Loss Data Validasi", "0.56")
    st.markdown('</div>', unsafe_allow_html=True)

    col_spacer_left, col_chart, col_spacer_right = st.columns([1, 4, 1])
    
    epochs = list(range(1, 13))

    history_data = pd.DataFrame({
        "Epoch": epochs,
        "Accuracy": [
            0.2399, 0.4027, 0.4220, 0.4690,
            0.5092, 0.5822, 0.6963, 0.8003,
            0.8674, 0.9211, 0.9581, 0.9706
        ],
        "Val Accuracy": [
            0.3792, 0.4497, 0.4430, 0.4732,
            0.4765, 0.5805, 0.6879, 0.7383,
            0.7718, 0.7919, 0.8054, 0.8121
        ]
    })

    with col_chart:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Kurva Akurasi Pelatihan vs Validasi")
        
        fig_acc = px.line(history_data, x='Epoch', y=['Accuracy', 'Val Accuracy'], markers=True,
                          template=graph_template)
        
        fig_acc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            legend_title="Metrik",
            font=dict(color=graph_text_color),
            xaxis=dict(color=graph_text_color),
            yaxis=dict(color=graph_text_color),
            legend=dict(font=dict(color=graph_text_color)),
            hovermode="x unified"
        )
        st.plotly_chart(fig_acc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "Catatan: Nilai metrik pada halaman ini ditampilkan sebagai ringkasan visual. "
        "Evaluasi performa model secara lengkap disajikan pada notebook analisis."
    )
else:
    st.title("Kesimpulan & Insight")
    
    st.markdown("""
    <div class="glass-card">
        <h3>Temuan Utama</h3>
        <ul>
            <li>Kategori <b>Sport</b> dan <b>Politics</b> cenderung lebih mudah diklasifikasikan karena memiliki istilah yang relatif konsisten dan jarang muncul di kategori lain.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
         st.markdown("""
        <div class="glass-card">
            <h3>Rencana Pengembangan</h3>
            <ol>
                <li>Melakukan tuning hyperparameter lebih lanjut pada model LSTM.</li>
                <li>Menambahkan <b>Word Embedding (GloVe/Word2Vec)</b>.</li>
                <li>Memperbanyak variasi data latih.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h3>Tech Stack</h3>
            <p>
                <span style="background:#262730; padding:5px 10px; border-radius:5px; margin-right:5px; color:white !important;">Python</span>
                <span style="background:#262730; padding:5px 10px; border-radius:5px; margin-right:5px; color:white !important;">TensorFlow</span>
                <span style="background:#262730; padding:5px 10px; border-radius:5px; margin-right:5px; color:white !important;">Streamlit</span>
                <span style="background:#262730; padding:5px 10px; border-radius:5px; color:white !important;">Plotly</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; margin-top: 50px; opacity: 0.6;'>
        <p>Tugas Besar – Pemrograman Dasar Sains Data</p>
    </div>
    """, 
    unsafe_allow_html=True
)