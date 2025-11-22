import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Text processing / stopwords
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
st.markdown("""
<style>
@media (max-width: 768px) {
    /* Font tetap kecil */
    .ag-cell, .ag-header-cell-label {
        font-size: 12px !important;
    }
    /* Biar bisa scroll horizontal */
    .ag-root-wrapper {
        overflow-x: auto !important;
    }
}
</style>
""", unsafe_allow_html=True)

# --- APP CONFIG ---
st.set_page_config(page_title="🩺 Edukasi Diabetes — Recommender", layout="wide")

# --- STOPWORDS SETUP ---
try:
    nltk.download('stopwords')
except:
    pass

try:
    nltk_stop = set(stopwords.words('indonesian'))
except:
    nltk_stop = set()

sastrawi_stop = set(StopWordRemoverFactory().get_stop_words())

custom_ui_stopwords = set([
    'meta','instagram','profile','profiles','more','also','messages','reply','see','follow',
    'likes','like','view','shared','sharedby','reels','story','stories','pm','dm','post','posts',
    'today','yesterday','ago','hrs','hours','d','h','m', 'translation', 'home', 'search', 
    'explore','notifications', 'create', 'api', 'non', 'english', 'contact', 'blog', 'jobs', 
    'job', 'help','privacy', 'articles', 'location', 'threads', 'uploading', 'users', 'verified',
    'start', 'app','conversation', 'locations', 'terms', 'lite', 'official', 'youtube', 'replies',
    'admin', 'aplikasi','scan', 'code', 'download', 'net', 'mail', 'edited', 'edit'
])

nama_umum = set([
    'ratna','anita','hasan','faisal','aman','siti','nur','agus','budi','andi','indah',
    'rini','dwi','yoga','ika','yuliana','aditya','setiawan','wulandari',
])

extra_stop = set([
    'agusafandiik07', 'ara20', 'amanpulungan','adik','akun','aja','aging','academy',
    'official','perkeni','klikdiabetes','sobatdiabet','halodoc','alodokter',
    'klikdokter','diabetesinitiative','adityaset1','allah','aplikasialodokter',
    'mganikcare','pbpersadia','ppperkeni','dengansatuklik','indonesia', 'ayat',
    'ahlinya', 'ahlinyagizi', 'akibatnya', 'akses', 'aktif', 'aplikasinya',
    'alat','american','anggap','angka','bacacaption'
])

bulan_stop = set([
    'januari','februari','maret','april','mei','juni','juli','agustus',
    'september','oktober','november','desember',
    'january','february','march','april','may','june','july',
    'august','september','october','november','december'
])

stopwords_all = nltk_stop | sastrawi_stop | ENGLISH_STOP_WORDS | custom_ui_stopwords | nama_umum | extra_stop | bulan_stop

# TOPIC DICTIONARY
topics_by_type = {
    "Edukasi Medis": ["gejala", "diabetes tipe", "insulin", "hipoglikemia", "hiperglikemia", "komplikasi", "gula darah", "autoimun", "antibodi", "kekurangan insulin", "onset muda", "ketoasidosis", "injeksi insulin", "pankreas", "sel beta", "komplikasi", "resiko", "edukasi", "diagnosis", "kontrol", "gestasional", "hamil", "autoantibody"],
    "Nutrisi": ["makan", "makanan", "diet", "kalori", "karbo", "gizi", "protein", "sehat", "pola makan", "menu", "menu sehat", "obesitas", "makanan bergizi", "nutrisi", "gemuk"],
    "Lifestyle": ["olahraga", "aktif", "jalan", "hidup sehat", "skrining", "manajemen gula", "gaya hidup", "pencegahan", "deteksi dini", "sehat", "stress", "senam", "jogging", "aktivitas"]
}

# LOAD DATA
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/AnggitaRisqiNC/content-personalization/refs/heads/main/multi_type_predictions.csv"
    response = requests.get(url)

    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        df = df.dropna(subset=["clean_caption_v2"]).reset_index(drop=True)
        return df
    else:
        st.error("Gagal memuat data dari GitHub 😭")
        return None
            
@st.cache_data
def fit_vectorizer(texts, max_features=5000, ngram=(1,2)):
    vec = TfidfVectorizer(stop_words=list(stopwords_all), max_features=max_features, ngram_range=ngram)
    X = vec.fit_transform(texts)
    return vec, X

# UI
st.header("🩺 Sistem Rekomendasi Konten Edukasi Diabetes")
st.sidebar.header("Isi data kamu dulu 😊")

nama = st.sidebar.text_input("Nama Lengkap")
usia = st.sidebar.number_input("Usia", min_value=1, max_value=120)
jenjang = st.sidebar.selectbox("Jenjang Pendidikan", ["SD", "SMP", "SMA/SMK", "D3", "S1", "S2", "S3"])
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
tipe_dm_label = st.sidebar.selectbox("Tipe Diabetes", ["Tipe 1", "Tipe 2", "Umum"])
lama_dm = st.sidebar.text_input("Lama menderita diabetes (misal: 2 tahun)")

# UI → CSV mapping
label_to_csv_value = {
    "Umum": "Umum",
    "Tipe 1": "Tipe1",
    "Tipe 2": "Tipe2"
}

tipe_dm_csv = label_to_csv_value[tipe_dm_label]

# LOAD TF-IDF
with st.spinner("Loading data & fitting TF-IDF..."):
    df = load_data()

    if df is None:
        st.stop()

    vectorizer, tfidf_matrix = fit_vectorizer(df["clean_caption_v2"].astype(str))

# TOP K SLIDER
top_k = st.sidebar.slider("Top K rekomendasi", 1, 100, 5)

# FUNCTION: dominant topic
def get_dominant_topic(text):
    text = text.lower()
    best_topic = "Tidak Teridentifikasi"
    best_score = 0

    for topic, keywords in topics_by_type.items():
        hits = sum(1 for k in keywords if k in text)
        if hits > best_score:
            best_score = hits
            best_topic = topic

    return best_topic

# BUTTON
if st.sidebar.button("✨ Tampilkan Rekomendasi"):

    if not nama:
        st.warning("Isi nama dulu ya 😅")

    else:
        score_map = {
            "Tipe1": "score_Tipe1",
            "Tipe2": "score_Tipe2",
            "Umum": "score_Umum"
        }

        score_col = score_map[tipe_dm_csv]

        df_filtered = df[df["predicted_type"] == tipe_dm_csv]
        df_filtered = df_filtered.sort_values(score_col, ascending=False)
        results = df_filtered.head(top_k).copy()

        results["topic_category"] = results["clean_caption_v2"].apply(get_dominant_topic)

        def truncate(text, max_chars=60):
            return text[:max_chars] + "..." if len(text) > max_chars else text

        table_data = pd.DataFrame({
            "No": range(1, len(results)+1),
            "📌 Akun": results["account"].fillna(""),
            "📖 Caption": results["clean_caption_v2"].fillna("").apply(truncate),
            "🔍 Kategori": results["topic_category"],
            "⭐ Skor": results[score_col].round(4),
            "🔗 Link": results["url"].fillna("")
        })

        st.session_state["table_data"] = table_data

        st.markdown(
            f'<h6 class="mobile-title">🎯 Hai Kak <b>{nama}</b>, berikut rekomendasi terbaik untuk kategori <b>{tipe_dm_label}</b></h6>',
            unsafe_allow_html=True
        )

# DISPLAY TABLE & FILTER
if "table_data" in st.session_state:

    table_data = st.session_state["table_data"].copy()

    # Select kategori berdasarkan kolom yang benar
    category_filter = st.selectbox(
        "Filter berdasarkan kategori topik",
        ["Semua"] + list(table_data["🔍 Kategori"].unique())
    )

    if category_filter != "Semua":
        table_data = table_data[table_data["🔍 Kategori"] == category_filter]

    if len(table_data) == 0:
        st.warning("Tidak ada konten yang cocok dengan filter 😭")

    else:
        link_renderer = JsCode("""
        class LinkCellRenderer {
            init(params) {
                this.eGui = document.createElement('a');
                this.eGui.href = params.value;
                this.eGui.target = '_blank';
                this.eGui.innerText = 'Klik IG';
                this.eGui.style.color = '#1DA1F2';
                this.eGui.style.textDecoration = 'none';
            }
            getGui() { return this.eGui; }
        }
        """)

        gb = GridOptionsBuilder.from_dataframe(table_data)
        gb.configure_default_column(resizable=True)
        gb.configure_column("No", width=60)
        gb.configure_column("📌 Akun", width=100)
        gb.configure_column(
            "📖 Caption",
            width=130,
            wrapText=True
        )
        gb.configure_column("🔍 Kategori", width=110)
        gb.configure_column("⭐ Skor", width=80)
        gb.configure_column("🔗 Link", cellRenderer=link_renderer, width=120)

        gridOptions = gb.build()

        AgGrid(
            table_data,
            gridOptions=gridOptions,
            fit_columns_on_grid_load=False,
            height=480,
            theme='alpine',
            allow_unsafe_jscode=True
        )










