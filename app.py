import nltk
import string
import pandas as pd
import gc
from random import sample
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
nltk.download('words')
words = set(nltk.corpus.words.words())
stopwords_ = set(stopwords.words('english'))
punctuation_ = set(string.punctuation)
stemmer = SnowballStemmer("english")

app = Flask(__name__)

df_reviews = None
df_users = None
df_business = None
df_tips = None
df_review_tips = None
current_user = None
current_user_id = ''


def filter_tokens(sent):
    return ([w for w in sent if not w in stopwords_ and not w in punctuation_])


def text_preprocess(string):
    word_list = []
    sent_tokens = sent_tokenize(string)
    tokens = [sent for sent in map(word_tokenize, sent_tokens)]

    tokens_lower = [[word.lower() for word in sent]
                    for sent in tokens]
    tokens_filtered = list(map(filter_tokens, tokens_lower))
    filtered_list = []
    for tlist in tokens_filtered:
        for w in tlist:
            if w != '...':
                filtered_list.append(stemmer.stem(w))

    # Remove Freq
    top_words = ['i', 'the', 'food', 'place', 'good', 'order', 'like', 'great', 'servic', 'it', 'time', 'we', 'one', 'get', 'go', 'tri', 'realli', 'restaur', 'would', 'friend', 'come', 'back', 'also', 'chicken', 'tast']
    for w in filtered_list:
        if w not in top_words:
            word_list.append(w)

    # Sampling
    sample_size = 1000
    if len(word_list) < 1000:
        sample_size = len(word_list)

    selected_word_list = sample(word_list, sample_size)

    return " ".join(selected_word_list)


def tfidf(corpus):
    # Transform in Tf-idf
    vectorizer_tfidf = TfidfVectorizer()

    corpus_tfidf = vectorizer_tfidf.fit_transform(corpus)
    return corpus_tfidf


def cosin_df():
    print("cosin_df starts")
    df_reviews_new = df_reviews[['business_id', 'text']].groupby('business_id')['text'].apply(' '.join).reset_index()
    df_tips_new = df_tips[['business_id', 'text']].groupby('business_id')['text'].apply(' '.join).reset_index()

    # Join two dfs of reviews and tips based on business_id
    global df_review_tips
    df_review_tips = pd.merge(df_reviews_new, df_tips_new, on=["business_id"], how='outer')
    df_review_tips['text'] = df_review_tips["text_x"] + " " + df_review_tips["text_y"]
    df_review_tips = df_review_tips[['business_id', 'text']]

    df_review_tips['cleaned_text'] = df_review_tips['text'].apply(lambda t: text_preprocess(str(t)))

    print("cosin_df ends")
    return df_review_tips


def cosin_recommender(category=None, user_requirement=None):
    print("cosin_recommender starts...")
    if user_requirement is not None:
        # Add user_requirements into reviews_tips_df
        df_review_tips_new = df_review_tips.append(
            {"business_id": "user_requirements", "cleaned_text": user_requirement}, ignore_index=True)

        corpus = df_review_tips_new['cleaned_text']
        # Text Transform
        corpus_tfidf = tfidf(corpus)
        doc_sim = cosine_similarity(corpus_tfidf, corpus_tfidf)

        # Add user_requirements into business_df
        df_business_new = df_business.append({"business_id": "user_requirements"}, ignore_index=True)

        business_index = df_business_new[df_business_new['business_id'] == "user_requirements"].index
        indices = doc_sim[business_index][0].argsort()[::-1][:11]

        suggest_list = []
        for ind in indices:
            if ind != business_index:
                suggest_list.append(df_business_new.iloc[ind]['name'])

        # Delete dataframes
        del df_review_tips_new
        del df_business_new
        gc.collect()

        print("cosin_recommender ends")
        return suggest_list
    else:
        if current_user is None:
            return []

        corpus = df_review_tips['cleaned_text']
        # Text Transform
        corpus_tfidf = tfidf(corpus)
        doc_sim = cosine_similarity(corpus_tfidf, corpus_tfidf)

        user_review = df_reviews[df_reviews['user_id'] == current_user_id].sort_values(['stars'], ascending=0).reset_index()
        selected_business_id = ''
        if category is None:
            user_top_star = user_review.iloc[0]['stars']
            user_top_review = user_review[user_review['stars'] == user_top_star]
            # Randomly pick a restaurant with with top ratings in the user's list
            user_top_review_sample = user_top_review.sample()
            if not user_top_review_sample.empty:
                selected_business_id = user_top_review_sample.iloc[0]['business_id']
            else:
                return []
        else:
            for _, row in user_review.iterrows():
                business_id = row["business_id"]
                business_category = df_business[df_business['business_id'] == business_id]['categories']
                if category in str(business_category):
                    selected_business_id = business_id
                    break

        print("cosin_recommender ends")
        if selected_business_id == '':
            return []
        else:
            business_index = df_business[df_business['business_id'] == selected_business_id].index
            indices = doc_sim[business_index][0].argsort()[::-1][:11]

            suggest_list = []
            for ind in indices:
                if ind != business_index:
                    suggest_list.append(df_business.iloc[ind]['name'])

            return suggest_list


def set_user(user_input):
    global current_user
    current_user = df_users[df_users['user_id'] == user_input]

    global current_user_id
    current_user_id = current_user['user_id'].to_string(index=False).strip()


@app.before_first_request
def preload_dataset():
    # transfer json file to dataframe
    global df_reviews
    df_reviews = pd.read_json("dataset/reviews.json", lines=True,
                              dtype={'review_id': str, 'user_id': str, 'business_id': str, 'stars': int,
                                     'date': str, 'text': str, 'useful': int, 'funny': int, 'cool': int})
    global df_users
    df_users = pd.read_json("dataset/users.json", lines=True,
                           dtype={'user_id': str, 'name': str, 'review_count': int, 'useful': int, 'funny': int,
                                  'cool': int, 'average_stars': float})
    global df_business
    df_business = pd.read_json("dataset/business.json", lines=True,
                               dtype={'name': str, 'review_count': int, 'business_id': str, 'address': str, 'city': str,
                                      'state': str, 'postal_code': str, 'latitude': float, 'longitude': float, 'stars': int,
                                      'is_open': int, 'attributes': dict, 'categories': str})
    global df_tips
    df_tips = pd.read_json("dataset/tips.json", lines=True,
                           dtype={'business_id': str, 'user_id': str, 'text': str})

    global df_review_tips
    df_review_tips = cosin_df()

    print("Dataset loaded")


@app.route('/', methods=['POST'])
def recommender():
    if request.method == "POST":
        user_input = request.data.decode("utf-8")
        print("user_input: " + str(user_input))
        if str(user_input).startswith('UserId'):
            userid = str(user_input).split(':')[1].strip()
            set_user(userid)
            if current_user is None:
                return 'User not found', 400
            else:
                user_name = current_user["name"].to_string(index=False)
                return 'User profile updated.\n Hello! ' + user_name, 200
        elif str(user_input).startswith('Category'):
            category = str(user_input).split(':')[1].strip()
            if 'Category: any' in str(user_input) or category == "":
                suggest_list = cosin_recommender()
            else:
                suggest_list = cosin_recommender(category)

            suggest_list_text = ", ".join(suggest_list) if len(suggest_list) > 0 else "None"
            result_text = "The recommended restaurants are: \n" + suggest_list_text
            return result_text, 200
        elif str(user_input).startswith('NewUser'):
            user_requirement = str(user_input).split(':')[1].strip()
            if user_requirement != "":
                suggest_list = cosin_recommender(None, user_requirement)
                suggest_list_text = ", ".join(suggest_list) if len(suggest_list) > 0 else "None"
                result_text = "Hello New User! \n" \
                              "The recommended restaurants are: \n" + suggest_list_text
                return result_text, 200
            else:
                return 'Invalid input. Please try again', 400
        else:
            return 'Invalid input. Please try again', 400


@app.route('/')
def index():
    return render_template("ui.html")


if __name__ == '__main__':
    app.run()
