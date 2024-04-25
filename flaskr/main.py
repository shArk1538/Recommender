from flask import (
    Blueprint, render_template, request
)

from .tools.data_tool import *   #加载了data_tool.py中的函数

from surprise import Reader
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity

import nltk
import string 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

stopword = stopwords.words('English')

bp = Blueprint('main', __name__, url_prefix='/')

movies, genres, rates = loadData()
# blogs, topics, rates = loadData()  # fixed：部分保持原变量名不变，保证前端的正常运行

@bp.route('/', methods=('GET', 'POST'))
def index():

    # Default Genres List
    default_genres = genres.to_dict('records')  

    # User Genres
    user_genres = request.cookies.get('user_genres')
    if user_genres:
        user_genres = user_genres.split(",")
    else:
        user_genres = []

    # User Rates
    user_rates = request.cookies.get('user_rates')
    if user_rates:
        user_rates = user_rates.split(",")
    else:
        user_rates = []

    # User Likes
    user_likes = request.cookies.get('user_likes')
    if user_likes:
        user_likes = user_likes.split(",")
    else:
        user_likes = []


    #函数在这一部分调用
    default_genres_movies = getMoviesByGenres(user_genres)[:-1]  #TODO: 展示电影类型的数量

    recommendations_movies, recommendations_message = getRecommendationBy(user_rates)

    likes_similar_movies, likes_similar_message = getLikedSimilarBy([int(numeric_string) for numeric_string in user_likes])

    likes_movies = getUserLikesBy(user_likes)

    return render_template('index.html',
                           genres=default_genres,
                           user_genres=user_genres,
                           user_rates=user_rates,
                           user_likes=user_likes,
                           default_genres_movies=default_genres_movies,
                           recommendations=recommendations_movies,
                           recommendations_message=recommendations_message,
                           likes_similars=likes_similar_movies,
                           likes_similar_message=likes_similar_message,
                           likes=likes_movies,
                           )  #与前端交互渲染网页


def getUserLikesBy(user_likes):
    results = []

    if len(user_likes) > 0:
        mask = movies['movieId'].isin([int(movieId) for movieId in user_likes])  #TODO
        results = movies.loc[mask]

        original_orders = pd.DataFrame()
        for _id in user_likes:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = movie
            else:
                original_orders = pd.concat([movie, original_orders])
        results = original_orders

    # return the result
    if len(results) > 0:
        return results.to_dict('records')
    return results


def getMoviesByGenres(user_genres):  # 用户选择的类型
    results = []

    # ====  Do some operations ====

    if len(user_genres) > 0:
        genres_mask = genres['id'].isin([int(id) for id in user_genres])
        user_genres = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres)
        user_genres_df.index = genres['name']
        movies_genres = movies.iloc[:, 5:]
        mask = (movies_genres.dot(user_genres_df) > 0).squeeze()
        results = movies.loc[mask][:30]

    # ==== End ====

    # return the result
    if len(results) > 0:
        return results.to_dict('records')
    return results


# Modify this function
def getRecommendationBy(user_rates):   # 用户打分
    results = []

    # ==== Do some operations ====

    # Check if there are any user_rates
    if len(user_rates) > 0:
        # Initialize a reader with rating scale from 1 to 5
        reader = Reader(rating_scale=(1, 5))

        # algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': True}) #TODO: 算法部分
        algo1 = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})  # user-based
        algo2 = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': False})  # item-based

        # Convert user_rates to rates from the user
        user_rates = ratesFromUser(user_rates)

        # Combine rates and user_rates into training_rates
        training_rates = pd.concat([rates, user_rates], ignore_index=True)

        # Load the training data from the training_rates DataFrame
        training_data = Dataset.load_from_df(training_rates, reader=reader)

        # Build a full training set from the training data
        trainset = training_data.build_full_trainset()

        # Fit the algorithm using the trainset
        algo1.fit(trainset)
        algo2.fit(trainset)

        ## Convert the raw user id to the inner user id using algo.trainset
        inner_id1 = trainset.to_inner_uid(5010) # user 序号
        inner_id2 = trainset.to_inner_iid(5010) # item 序号

        ## Get the nearest neighbors of the inner_id
        user_neighbors = algo1.get_neighbors(inner_id1, k=3)   #TODO: K近邻数量
        item_neighbors = algo2.get_neighbors(inner_id2, k=3)

        ## Convert the inner user ids of the neighbors back to raw user ids
        neighbors_uid = [algo1.trainset.to_raw_uid(x) for x in user_neighbors]
        neighbors_iid = [algo2.trainset.to_raw_uid(x) for x in item_neighbors]

        ## Filter out the movies this neighbor likes.
        results_movies1 = rates[rates['userId'].isin(neighbors_uid)]  # 获取到近邻的结果
        results_movies2 = rates[rates['movieId'].isin(neighbors_iid)]

        moviesIds1 = results_movies1[results_movies1['rating'] > 3.5]['movieId']
        moviesIds2 = results_movies2[results_movies2['rating'] > 3.5]['movieId']

        # Convert the movie ids to details.
        results = movies[movies['movieId'].isin(moviesIds1)][:12]
        # results = movies[movies['movieId'].isin(moviesIds2)][:12]
        

    # Return the result
    if len(results) > 0:
        return results.to_dict('records'), "These blogs are similar to your liked."
    return results, "No similar blog found."
    # ==== End ====


# Modify this function
def getLikedSimilarBy(user_likes):
    results = []

    # ==== Do some operations ====
    if len(user_likes) > 0:

        blog_TF_IDF_vector, tfidf_feature_list = build_tfidf_vectors()
        
        user_profile = build_tfidf_user_profile(user_likes, blog_TF_IDF_vector, tfidf_feature_list, normalized=True)
        
        results = generate_tf_idf_recommendation_results(user_profile, blog_TF_IDF_vector, tfidf_feature_list, k=12)

        # # Step 1: Representing items with one-hot vectors
        # item_rep_matrix, item_rep_vector, feature_list = item_representation_based_movie_genres(movies)

        # # Step 2: Building user profile
        # user_profile = build_user_profile(user_likes, item_rep_vector, feature_list)

        # # Step 3: Predicting user interest in items
        # results = generate_recommendation_results(user_profile, item_rep_matrix, item_rep_vector, 12)

    # Return the result
    if len(results) > 0:
        return results.to_dict('records'), "The movies are similar to your liked blogs."
    return results, "No similar blogs found."

    # ==== End ====

##==============================content-based recommand=========================
# import nltk
# import string 
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
# from nltk.stem.wordnet import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# stopword = stopwords.words('English')


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    

# def prepocessing(text):
#     # lower case
#     text = str(text).lower()
    
#     # remove punctuation
#     text_rp = "".join([char for char in text if char not in string.punctuation])
    
#     # word tokenization 
#     tokens = word_tokenize(text_rp)
    
#     # remove stopwords  
    
#     tokens_without_stopwords = [word for word in tokens if word not in stopword]

#     # lemm
#     tagged_tokens = nltk.pos_tag(tokens_without_stopwords)
#     #print(tagged_tokens)
#     tokens_processed = []
    
#     lemmatizer = WordNetLemmatizer()
#     for word, tag in tagged_tokens:
#         word_net_tag = get_wordnet_pos(tag)
#         if word_net_tag != '':
#             tokens_processed.append(lemmatizer.lemmatize(word, word_net_tag))
#         else:
#             tokens_processed.append(word)
#     text_processed = ' '.join(tokens_processed)
    
#     return text_processed

from nltk.stem import PorterStemmer
import re

def preprocessing(text, flg_stemm=False, flg_lemm=True, lst_stopwords=stopword):
    text=str(text).lower()
    text=text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    lst_text = text.split()
    if lst_stopwords is not None:
        lst_text=[word for word in lst_text if word not in lst_stopwords]
    if flg_lemm:
        lemmatizer = WordNetLemmatizer()
        lst_text = [lemmatizer.lemmatize(word) for word in lst_text]
    if flg_stemm:
        stemmer = PorterStemmer()
        lst_text = [stemmer.stem(word) for word in lst_text]
    text=" ".join(lst_text)
    return text


def build_tfidf_vectors():
    blog_vectors = movies.copy(deep=True)
    blog_vectors['content'] = blog_vectors['content'].fillna('')
    
    from sklearn.feature_extraction.text import TfidfVectorizer

    #Define a TF-IDF Vectorizer Object. 
    tfidf = TfidfVectorizer(
        preprocessor=preprocessing, 
        ngram_range=(1,1),
        max_features=30)

    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(movies['content'])

    blog_TF_IDF_vector = pd.DataFrame(tfidf_matrix.toarray(),columns=tfidf.get_feature_names_out())
    blog_TF_IDF_vector['movieId'] = movies['movieId']

    return blog_TF_IDF_vector, tfidf.get_feature_names_out()[0:30]

def build_tfidf_user_profile(user_likes, blog_TF_IDF_vector, tfidf_feature_list, normalized=True):
    user_blog = blog_TF_IDF_vector[blog_TF_IDF_vector['movieId'].isin(user_likes)]

    user_blog_df = user_blog[tfidf_feature_list].mean()

    user_profile = user_blog_df.T

    if normalized:
        user_profile = user_profile / sum(user_profile.values)
    
    return user_profile


def generate_tf_idf_recommendation_results(user_profile, blog_TF_IDF_vector, tfidf_feature_list, k=12):
    u_v = user_profile
    u_v_matrix = [u_v]

    recommendation_table =  cosine_similarity(u_v_matrix, blog_TF_IDF_vector[tfidf_feature_list])

    recommendation_table_df = movies.copy(deep=True)  # TODO
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]

    return rec_result



##====================================================================================
# # getLikedSimilarBy函数嵌套的三个函数
# def item_representation_based_movie_genres(movies_df):
#     '''
#     Build One-hot table for the movies' list
#     '''
#     movies_with_genres = movies_df.copy(deep=True)

#     genre_list = movies_with_genres.columns[5:]
#     movies_genre_matrix = movies_with_genres[genre_list].to_numpy()
#     return movies_genre_matrix, movies_with_genres, genre_list


# def build_user_profile(movieIds, item_rep_vector, feature_list, normalized=True):
#     '''
#     Build user profile
#     '''
#     ## Calculate item representation matrix to represent user profiles
#     user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(movieIds)]
#     user_movie_df = user_movie_rating_df[feature_list].mean()
#     user_profile = user_movie_df.T

#     if normalized:
#         user_profile = user_profile / sum(user_profile.values)

#     return user_profile


# def generate_recommendation_results(user_profile,item_rep_matrix, movies_data, k=12):

#     u_v = user_profile.values
#     u_v_matrix = [u_v]

#     # Comput the cosine similarity
#     recommendation_table = cosine_similarity(u_v_matrix, item_rep_matrix)

#     recommendation_table_df = movies_data.copy(deep=True)
#     recommendation_table_df['similarity'] = recommendation_table[0]
#     rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]

#     return rec_result