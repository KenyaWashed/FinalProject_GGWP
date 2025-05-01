from multiprocessing import dummy
import pandas as pd
import joblib
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def get_recommendation(original_df, similarity_df, input_budget, input_language, input_genres, input_keywords, input_runtime, input_companies, input_credits):
    '''
    Parameters:
    - original_df: datase after preprocessing
    - similarity_df: the similarity.csv file
    - input_budget(float): Note: Đưa input đã chuẩn hóa vào
    - input_language(str)
    - input_genres(list): Note: Đưa vào một list, tách dữ liệu ra trước
    - input_keywords(list): Note: Đưa vào một list, tách dữ liệu ra trước
    - input_runtime(float): Note: Đưa input đã chuẩn hóa vào
    - input_companies(list): Note: Đưa vào một list, tách dữ liệu ra trước
    - input_credits(list): Note: Đưa vào một list, tách dữ liệu ra trước
    
    Return:
    Trả về một dict, vd:
    {
        'runtime_suggestion': 'Runtime của bạn hơi cao (120 phút). Nên giảm xuống khoảng 100 phút.',
        'company_suggestion': 'Cân nhắc hợp tác với các hãng sản xuất sau: Warner Bros, Universal Pictures.',
        'actor_suggestion': 'Có thể mời các diễn viên nổi bật như: Tom Hanks, Robert Downey Jr.',
        'needed_change': 3
    }
    need_change chỉ để ra tín hiệu, nếu bằng 0 thì không có gợi ý gì hết
    '''
    
    
    similarity_vector = create_similarity_vector(input_budget, input_language, input_genres, input_keywords)
    
    top_n_indices = get_top_similar_movie(similarity_vector, similarity_df, n=50)
    
    recommendation = recommend_changes(original_df, top_n_indices, input_runtime, input_companies, input_credits)
    
    return recommendation



# This function took the input from the user for some features, then return a vector (dataframe) that suited for cosine similarity
def create_similarity_vector(budget, original_language, genres, keywords):
    language_encoder = joblib.load('assets/language_encoder.pkl')
    genres_binarizer = joblib.load('assets/genres_binarizer.pkl')
    keywords_binarizer = joblib.load('assets/keywords_binarizer.pkl')
    
    # Note: Order: budget -> genres -> keywords -> original_language
    
    if not isinstance(genres, list):
        genres = [genres]
    if not isinstance(keywords, list):
        keywords = [keywords]
        
    # Genres
    genres_encoded = genres_binarizer.transform([genres])  # cần dạng 2D: [genres]
    genres_df = pd.DataFrame(
        genres_encoded,
        columns=[f"genres_{name}" for name in genres_binarizer.classes_]
    )

    # Keywords
    keywords_encoded = keywords_binarizer.transform([keywords])
    keywords_df = pd.DataFrame(
        keywords_encoded,
        columns=[f"keywords_{name}" for name in keywords_binarizer.classes_]
    )
    
    # Language
    language_encoded = language_encoder.transform([[original_language]]).toarray()
    language_df = pd.DataFrame(
        language_encoded,
        columns=[f"language_{name}" for name in language_encoder.categories_[0]]
    )
    
    # Budget
    budget_df = pd.DataFrame([[budget]], columns=['budget'])
    
    
    final_vector = pd.concat([budget_df, genres_df, keywords_df, language_df], axis=1)
    
    return final_vector


# This function will find return the index of the most similar vectors in the similarity_df
# Note: The index of the similarity_df is the movie id
def get_top_similar_movie(vector, similarity_df, n=10):
    similarity_scores = cosine_similarity(vector, similarity_df)[0]
    
    similarity_serie = pd.Series(similarity_scores, index=similarity_df.index)
    similarity_serie = similarity_serie.sort_values(ascending=False)
    top_n_indices = similarity_serie.head(n).index
    
    return top_n_indices


def recommend_changes(original_df, top_n_indices, input_runtime, input_companies, input_credits):
    # Get the similar movies
    similar_movies = original_df.loc[top_n_indices]
    n = 10
    top_n_movies = similar_movies.sort_values(by='revenue', ascending=False).head(n)
    needed_change = 3
    
    # ===== Runtime =====
    recommended_runtime = top_n_movies['runtime'].median()
    
    # Turn the runtime back into original scale
    scaler = joblib.load('assets/runtime_scaler.pkl')
    dummy = np.zeros((1, 6))
    dummy[0, 3] = recommended_runtime
    recommended_runtime = scaler.inverse_transform(dummy)[0][3]
    
    
    
    if input_runtime > recommended_runtime + 20:
        runtime_suggestion = f"Runtime của bạn hơi cao ({int(input_runtime)} phút). Nên giảm xuống khoảng {int(recommended_runtime)} phút."
    elif input_runtime < recommended_runtime - 20:
        runtime_suggestion = f"Runtime của bạn hơi thấp ({int(input_runtime)} phút). Nên tăng lên khoảng {int(recommended_runtime)} phút."
    else:
        needed_change -= 1
        runtime_suggestion = f'Runtime của bạn đã ổn.'

    # ===== Production Companies =====
    all_companies = []
    for companies_str in top_n_movies['production_companies']:
        if pd.notna(companies_str):
            all_companies.extend(companies_str.split('-'))
    company_counter = Counter(all_companies)
    most_common_companies = [company for company, _ in company_counter.most_common()]
    recommended_companies = random.sample(
    [c for c in most_common_companies if c not in input_companies][:10], 
    k=min(3, len([c for c in most_common_companies if c not in input_companies][:10]))
)
    if recommended_companies:
        company_suggestion = f"Cân nhắc hợp tác với các hãng sản xuất sau: {', '.join(recommended_companies)}."
    else:
        needed_change -= 1
        company_suggestion = None

    # ===== Credits (Actors) =====
    all_actors = []
    for credits_str in top_n_movies['credits']:
        if pd.notna(credits_str):
            all_actors.extend(credits_str.split('-'))
    actor_counter = Counter(all_actors)
    most_common_actors = [actor for actor, _ in actor_counter.most_common()]
    recommended_actors = random.sample([a for a in most_common_actors if a not in input_credits][:10], 
    k=min(3, len([a for a in most_common_actors if a not in input_credits][:10])))

    if recommended_actors:
        actor_suggestion = f"Có thể mời các diễn viên nổi bật như: {', '.join(recommended_actors)}."
    else:
        needed_change -= 1
        actor_suggestion = None


    return {
        'runtime_suggestion': runtime_suggestion,
        'company_suggestion': company_suggestion,
        'actor_suggestion': actor_suggestion,
        'needed_change': needed_change
    }