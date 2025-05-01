import streamlit as st
import pandas as pd
import numpy as np
import pickle
from recommend_generator_v1 import get_recommendation

# === Load mô hình và dữ liệu ===
model = pickle.load(open('rfr_best_model.pkl', 'rb'))

# Nếu cần, load thêm original_df và similarity_df ở đây
# original_df = pd.read_csv("preprocessed_movies.csv")
# similarity_df = pd.read_csv("similarity.csv")

# === Giao diện 2 tab ===
tab1, tab2 = st.tabs(["📈 Dự đoán doanh thu", "💡 Gợi ý cải thiện"])

# === Tab 1: Dự đoán doanh thu ===
with tab1:
    st.title("🎬 Feature Sugestion")

    title = st.text_input("Tên phim", "Example Movie Title")
    genre = st.selectbox("Thể loại", ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi'])
    production_company = st.text_input("Hãng sản xuất", "Example Production Company")
    language = st.selectbox("Ngôn ngữ gốc", ['en', 'fr', 'es', 'zh', 'hi'])
    runtime = st.slider("Thời lượng phim (phút)", 60, 240, 120)
    budget = st.number_input("Ngân sách (USD)", min_value=10000, max_value=500_000_000, value=50_000_000, step=1_000_000)

    # === Tạo input vector ===
    feature_names = model.feature_names_in_
    data = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)

    data['originally_english'] = 1 if language == "en" else 0
    data['log_runtime_processed'] = np.log1p(runtime)
    data['log_budget_processed'] = np.log1p(budget)

    top_studios = ['Warner Bros.', 'Universal Pictures', 'Walt Disney Pictures', '20th Century Fox', 'Paramount Pictures']
    data['topStudio'] = 1 if production_company in top_studios else 0

    top_genre_dict = {'Action': 1, 'Comedy': 2, 'Drama': 3, 'Horror': 4, 'Romance': 5, 'Sci-Fi': 6}
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
        st.success(f"💰 Dự đoán doanh thu: **${revenue[0]:,.2f} USD**")

# === Tab 2: Gợi ý cải thiện ===
with tab2:
    st.title("💡 Gợi ý cải thiện phim")

    st.info("Nhập thông tin để nhận các gợi ý tối ưu hoá cho phim")

    budget_norm = np.log1p(budget)
    runtime_norm = np.log1p(runtime)
    genres_input = [genre]
    keywords_input = []  # Nếu có thể nhập từ người dùng thì thêm vào
    companies_input = [production_company]
    credits_input = []  # Nếu có tên diễn viên thì thêm vào

    # Dummy DataFrame placeholder (nếu chưa load được file thật)
    import pandas as pd
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
            st.success("✅ Phim của bạn đã tối ưu. Không cần thay đổi thêm.")
        else:
            if 'runtime_suggestion' in recommendation:
                st.warning("⏱ " + recommendation['runtime_suggestion'])
            if 'company_suggestion' in recommendation:
                st.info("🏢 " + recommendation['company_suggestion'])
            if 'actor_suggestion' in recommendation:
                st.info("🎭 " + recommendation['actor_suggestion'])
