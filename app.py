import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
import re

# --- PRE-REQUISITES ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="🩺 Edukasi Diabetes — Recommender", layout="wide")

# Custom CSS
st.markdown("""
<style>
.stApp { background-color: #e0e0e0; }
[data-testid="stSidebar"] { background-color: #d6d6d6; }
.ag-theme-alpine .ag-header { background-color: #FFC0CB !important; }
.ag-theme-alpine .ag-header-cell { background-color: #FF69B4 !important; }
.ag-theme-alpine .ag-header-cell-label { color: white !important; font-weight: bold !important; }
.warning-box { background-color: #FBEC5D; padding: 10px; border-radius: 5px; color: #000000; font-weight: bold; margin: 10px 0px; }
</style>
""", unsafe_allow_html=True)

# --- 2. QUERY REFERENSI & KAMUS TOPIK ---
queries = {
    "Edukasi Tipe 1": "diabetes tipe 1 tipe 1 type 1 dm tipe 1 juvenile autoimun sel beta antibodi genetik suntik insulin injeksi insulin jarum insulin insulin pump ketoasidosis dka keton hipoglikemia gula darah rendah",
    "Edukasi Tipe 2": "diabetes tipe 2 dm tipe 2 type 2 tipe 2 resistensi insulin obat oral minum obat metformin glibenklamid luka kaki ulkus kaki diabetes kaki busuk gangren amputasi leher hitam acanthosis otak polusi kekuatan otot hipertensi keturunan genetik",
    "Edukasi Umum": "diabetes kencing manis sakit gula gula darah cek gula kadar gula hba1c glukosa skrining medical karbohidrat check up gejala tanda sering kencing haus terus cepat lapar kesemutan kebas mata kabur pola makan makanan sehat kurangi gula diet sehat minuman manis boba teh manis makanan olahan junk food obesitas kegemukan berat badan turun berat buncit olahraga senam jalan kaki sepeda aktivitas fisik hidup sehat gaya hidup stress tidur cukup edukasi tips sehat kata dokter cegah diabetes"
}

topics_keywords = {
    "Edukasi Medis": ["insulin", "obat", "metformin", "glukosa", "hba1c", "diagnosis", "gejala", "komplikasi", "tipe", "medis"],
    "Nutrisi": ["makan", "diet", "karbohidrat", "gula", "kalori", "nutrisi", "buah", "sayur", "protein", "lemak"],
    "Lifestyle": ["olahraga", "jalan", "senam", "stress", "tidur", "aktivitas", "fisik", "sehat", "hidup", "gaya"]
}

# --- 3. LOAD DATA ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/AnggitaRisqiNC/content-personalization/refs/heads/main/data_postdiabetes.csv"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        # Pastikan kita pakai kolom yang sudah di-stemming
        df = df.dropna(subset=["clean_caption_stemmed"]).reset_index(drop=True)
        return df
    return None

df = load_data()

# --- 4. SIDEBAR INPUT ---
st.header("🩺 Sistem Rekomendasi Konten Edukasi Diabetes")
st.sidebar.header("Isi data kamu dulu 😊")

nama = st.sidebar.text_input("Nama Kamu")
usia = st.sidebar.number_input("Usia Kamu", 1, 100, 25)
tipe_dm_label = st.sidebar.selectbox("Pilih Kategori Diabetes", list(queries.keys()))
top_k = st.sidebar.slider("Top K rekomendasi", 1, 20, 5)

st.sidebar.markdown('<div class="warning-box">⚠️ Hasil rekomendasi ini berbasis kemiripan teks (Cosine Similarity) dan bukan merupakan saran medis resmi</div>', unsafe_allow_html=True)

def identify_topic(text):
    text = str(text).lower()
    for topic, keywords in topics_keywords.items():
        if any(word in text for word in keywords): return topic
    return "Lainnya"

def clean_for_human(text):
    text = str(text)
    
    lines = text.split('\n')
    if len(lines) > 1:
        text = " ".join(lines[1:])
    text = re.sub(r'\w+\s•\s\w+', '', text)
    text = re.sub(r'\d+[wmhd]\s', '', text)
    
    trash = ["See translation", "View all", "comments", "Profile", "Follow", "likes"]
    for word in trash:
        text = text.replace(word, "")
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def ai_summarize(text, sentence_count=1):
    if len(str(text)) > 200:
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("indonesian"))
            summarizer = LexRankSummarizer()
            summary = summarizer(parser.document, sentence_count)
            return " ".join([str(s) for s in summary])
        except: return str(text)[:150] + "..."
    return text

# --- 6. PROSES HITUNG COSINE SIMILARITY ---
if st.sidebar.button("✨ Tampilkan Rekomendasi", type="primary"):
    if not nama:
        st.warning("Isi nama dulu ya 😅")
    else:
        # Gabungkan Query dengan Dataset
        query_text = queries[tipe_dm_label]
        all_texts = df['clean_caption_stemmed'].tolist()
        all_texts.append(query_text)

        # Fit TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Hitung Similarity (Query vs Semua Dokumen)
        cosine_sim = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()

        # Ambil hasil
        df['similarity_score'] = cosine_sim
        df['topic_category'] = df['clean_caption_stemmed'].apply(identify_topic)

        # Sort & Filter Top-K
        results = df.sort_values('similarity_score', ascending=False).head(top_k)
        results['display_caption'] = results['caption'].apply(clean_for_human)
        results['caption_ringkas'] = results['display_caption'].apply(lambda x: ai_summarize(x, 1))

        # Simpan ke session state agar filter topik bisa jalan
        st.session_state['results'] = results
        st.session_state['user_name'] = nama

# --- 7. TAMPILAN TABEL REKOMENDASI ---
if 'results' in st.session_state:
    res = st.session_state['results']
    st.success(f"🎯 Hai {st.session_state['user_name']}, ini konten yang paling mirip dengan {tipe_dm_label}")
  
    # Tambahan Filter Topik (Nutrisi, Lifestyle, Medis)
    filter_topik = st.selectbox("Saring berdasarkan Topik:", ["Semua", "Nutrisi", "Lifestyle", "Edukasi Medis"])
    display_df = res.copy()
    if filter_topik != "Semua":
        display_df = display_df[display_df['topic_category'] == filter_topik]

    if display_df.empty:
        st.info("Wah, tidak ada konten dengan topik tersebut di Top-K ini.")
    else:
        # Setup AgGrid
        link_renderer = JsCode("""
        class LinkCellRenderer {
            init(params) {
                this.eGui = document.createElement('a');
                this.eGui.href = params.value;
                this.eGui.target = '_blank';
                this.eGui.innerText = '🔗 Buka IG';
                this.eGui.style.color = '#FF69B4';
                this.eGui.style.fontWeight = 'bold';
                this.eGui.style.textDecoration = 'none';
            }
            getGui() { return this.eGui; }
        }
        """)

        # Pilih kolom dan kasih header nama pakai emoji
        gb = GridOptionsBuilder.from_dataframe(display_df[['account', 'caption_ringkas', 'topic_category', 'similarity_score', 'url']])
        
        # Pengaturan Header dengan Emoji
        gb.configure_column("account", headerName="👤 Akun Instagram")
        gb.configure_column("caption_ringkas", headerName="📝 Inti Konten", wrapText=True, autoHeight=True, width=400)
        gb.configure_column("topic_category", headerName="🏷️ Topik")
        gb.configure_column("similarity_score", 
                            headerName="📊 Skor Sim", 
                            valueFormatter="x.format('.4f')",
                            sort="desc") # Biar yang paling mirip otomatis di atas
        gb.configure_column("url", 
                            headerName="🌐 Link Post", 
                            cellRenderer=link_renderer)

        gridOptions = gb.build()

        # Tampilkan Tabel
        AgGrid(display_df, 
               gridOptions=gridOptions, 
               allow_unsafe_jscode=True, 
               theme='alpine',
               fit_columns_on_grid_load=True)








