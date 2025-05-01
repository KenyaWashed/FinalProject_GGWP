import streamlit as st
import pandas as pd
import numpy as np
import pickle
from recommend_generator_v1 import get_recommendation

st.set_page_config(layout="wide")

# === Thanh menu ngang tùy chỉnh ===
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://pickamovieforme.b-cdn.net/wp-content/uploads/2020/06/bg-tablet2.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        
        /* Custom navbar styling */
        .custom-navbar {
            height: 65px;
            display: flex;
            align-items: center;
            padding: 0 40px;
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: transparent; /* Bỏ background */
        }
        
        .custom-navbar .menu-left {
            display: flex;
            align-items: center;
        }
        
        .custom-navbar .menu-right {
            display: flex;
            margin-left: auto;
            align-items: center;
        }
        
        .custom-navbar a {
            color: white;
            text-decoration: none;
            margin-right: 40px;
            font-weight: bold;
            font-size: 16px;
        }
        
        .custom-navbar a:hover {
            color: #ff4b4b;
        }
        
        /* Make text more visible */
        h1, h2, h3, p, label {
            color: white !important;
            text-shadow: 1px 1px 3px black;
            font-weight: bold !important;
        }
        
        /* Style for the title */
        .title-section {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        /* Style for the inputs */
        .stTextInput, .stSelectbox, .stSlider, .stNumberInput {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 15px;
        }
        
        /* Style for tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 10px 10px 0 0;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: white;
            font-weight: bold;
        }
        
        /* Style for tab content */
        .stTabs [data-baseweb="tab-panel"] {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 0 0 10px 10px;
            padding: 20px;
        }
        
        /* Style for buttons */
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
        }
        
        .stButton > button:hover {
            background-color: #ff2a2a;
        }
    </style>

    <div class="custom-navbar">
        <div class="menu-left">
            <img src="https://cdn-icons-png.flaticon.com/256/9054/9054897.png" alt="logo" style="height: 50px; margin-right: 20px;">
        </div>
        <div class="menu-right">
            <a href="#">MOVIE PICKER</a>
            <a href="#">TOP GENRES</a>
            <a href="#">TOP ACTORS</a>
            <a href="#">BLOG</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# === Nội dung bên dưới ===
st.markdown('<div class="title-section">', unsafe_allow_html=True)
st.title("🎬 Movie Revenue Predictor")
st.markdown('</div>', unsafe_allow_html=True)

# === Load mô hình và dữ liệu ===
model = pickle.load(open('model/rfr_best_model.pkl', 'rb'))

# === Giao diện 2 tab ===
tab1, tab2 = st.tabs(["📈 Dự đoán doanh thu", "💡 Gợi ý cải thiện"])

# === Tab 1: Dự đoán doanh thu ===
with tab1:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.header("📊 Nhập thông tin để dự đoán doanh thu phim")

    title = st.text_input("Tên phim", "Example Movie Title")
    genre = st.selectbox(
        "Thể loại", ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi'])
    production_company = st.text_input(
        "Hãng sản xuất", "Example Production Company")
    language = st.selectbox("Ngôn ngữ gốc", ['en', 'fr', 'es', 'zh', 'hi'])
    runtime = st.slider("Thời lượng phim (phút)", 60, 240, 120)
    budget = st.number_input("Ngân sách (USD)", min_value=10000,
                             max_value=500_000_000, value=50_000_000, step=1_000_000)

    # === Tạo input vector ===
    feature_names = model.feature_names_in_
    data = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)

    data['originally_english'] = 1 if language == "en" else 0
    data['log_runtime_processed'] = np.log1p(runtime)
    data['log_budget_processed'] = np.log1p(budget)

    top_studios = ['Warner Bros.', 'Universal Pictures',
                   'Walt Disney Pictures', '20th Century Fox', 'Paramount Pictures']
    data['topStudio'] = 1 if production_company in top_studios else 0

    top_genre_dict = {'Action': 1, 'Comedy': 2,
                      'Drama': 3, 'Horror': 4, 'Romance': 5, 'Sci-Fi': 6}
    genre_rank = top_genre_dict.get(genre, 10)
    data['log_genre_rank'] = np.log1p(genre_rank)
    data['log_genres_no'] = np.log1p(1)

    data['vote_count'] = 0
    data['log_vote_average'] = 0
    data['log_num_credits'] = 0
    data['log_numTopActors'] = 0
    data['log_numTopStudios'] = 0
    data['log_num_studios'] = 0
    data['log_studioRank'] = 0
    data['log_budget_to_year_ratio'] = data['log_budget_processed']
    data['log_runtime_to_year_ratio'] = data['log_runtime_processed']
    data['log_title_length'] = np.log1p(len(title.split()))
    data['has_tagline'] = 0
    data['topLeadActor'] = 0
    data['has_keywords'] = 0
    data['has_poster_path'] = 0
    data['has_backdrop_path'] = 0
    data['fridayRelease'] = 1
    data['Summer'] = 1

    for col in feature_names:
        if col not in data.columns:
            data[col] = 0

    if st.button("🎯 Dự đoán doanh thu"):
        log_revenue = model.predict(data)
        revenue = np.expm1(log_revenue)
        st.markdown(f'<div style="background-color: rgba(25, 135, 84, 0.7); padding: 15px; border-radius: 5px; margin-top: 20px;"><h3 style="color: white; margin: 0;">💰 Dự đoán doanh thu: <b>${revenue[0]:,.2f} USD</b></h3></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# === Tab 2: Gợi ý cải thiện ===
with tab2:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.header("💡 Gợi ý cải thiện phim")
    st.markdown('<div style="background-color: rgba(13, 110, 253, 0.7); padding: 15px; border-radius: 5px; margin-bottom: 20px;"><p style="color: white; margin: 0;">Nhập thông tin để nhận các gợi ý tối ưu hoá cho phim</p></div>', unsafe_allow_html=True)

    budget_norm = np.log1p(budget)
    runtime_norm = np.log1p(runtime)
    genres_input = [genre]
    keywords_input = []
    companies_input = [production_company]
    credits_input = []

    original_df = pd.read_csv("data/processed_movies.csv")
    similarity_df = pd.read_csv("data/similarity.csv")

    if st.button("🔍 Xem gợi ý cải thiện"):
        recommendation = get_recommendation(
            original_df=original_df,
            similarity_df=similarity_df,
            input_budget=budget_norm,
            input_language=language,
            input_genres=genres_input,
            input_keywords=keywords_input,
            input_runtime=runtime_norm,
            input_companies=companies_input,
            input_credits=credits_input,
        )

        if recommendation['needed_change'] == 0:
            st.markdown('<div style="background-color: rgba(25, 135, 84, 0.7); padding: 15px; border-radius: 5px; margin-top: 20px;"><p style="color: white; margin: 0;">✅ Phim của bạn đã tối ưu. Không cần thay đổi thêm.</p></div>', unsafe_allow_html=True)
        else:
            suggestion_html = '<div style="background-color: rgba(255, 193, 7, 0.7); padding: 15px; border-radius: 5px; margin-top: 20px;">'
            
            if 'runtime_suggestion' in recommendation:
                suggestion_html += f'<p style="color: white; margin-bottom: 10px;">⏱ {recommendation["runtime_suggestion"]}</p>'  
            
            if 'company_suggestion' in recommendation:
                suggestion_html += f'<p style="color: white; margin-bottom: 10px;">🏢 {recommendation["company_suggestion"]}</p>'  
            
            if 'actor_suggestion' in recommendation:
                suggestion_html += f'<p style="color: white; margin-bottom: 0px;">🎭 {recommendation["actor_suggestion"]}</p>'  
            
            suggestion_html += '</div>'
            st.markdown(suggestion_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
