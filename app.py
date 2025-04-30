import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === Load mô hình ===
model = pickle.load(open('model/rfr_best_model.pkl', 'rb'))

st.title("🎬 Movie Revenue Predictor")

# === Input từ người dùng ===
title = st.text_input("Tên phim", "Example Movie Title")

genres = st.multiselect(
    "Thể loại (có thể chọn nhiều)", 
    ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi'], 
    default=['Action']
)

production_company = st.text_input("Hãng sản xuất", "Example Production Company")

language = st.selectbox("Ngôn ngữ gốc", ['en', 'fr', 'es', 'zh', 'hi'])

runtime = st.slider("Thời lượng phim (phút)", 60, 240, 120)

budget = st.number_input("Ngân sách (USD)", min_value=10000, max_value=500_000_000, value=50_000_000, step=1_000_000)

# === Tạo input vector ===
feature_names = model.feature_names_in_
data = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)

# === Thiết lập các đặc trưng từ input ===
data['originally_english'] = 1 if language == "en" else 0
data['log_runtime_processed'] = np.log1p(runtime)
data['log_budget_processed'] = np.log1p(budget)

# Studio đặc trưng
top_studios = ['Warner Bros.', 'Universal Pictures', 'Walt Disney Pictures', '20th Century Fox', 'Paramount Pictures']
data['topStudio'] = 1 if production_company in top_studios else 0

# Genre rank & số lượng
top_genre_dict = {'Action': 1, 'Comedy': 2, 'Drama': 3, 'Horror': 4, 'Romance': 5, 'Sci-Fi': 6}
genre_ranks = [top_genre_dict.get(g, 10) for g in genres]
if genre_ranks:
    data['log_genre_rank'] = np.log1p(min(genre_ranks))  # hoặc np.mean(...) nếu bạn muốn
    data['log_genres_no'] = np.log1p(len(genre_ranks))
else:
    data['log_genre_rank'] = np.log1p(10)
    data['log_genres_no'] = np.log1p(0)

# Một số đặc trưng mặc định
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

# Đảm bảo đủ cột
for col in feature_names:
    if col not in data.columns:
        data[col] = 0

# === Dự đoán ===
if st.button("🎯 Dự đoán doanh thu"):
    log_revenue = model.predict(data)
    revenue = np.expm1(log_revenue)
    st.success(f"💰 Dự đoán doanh thu: **${revenue[0]:,.2f} USD**")
