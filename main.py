# import nltk
# import string
# import pandas as pd
# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('words')
# words = set(nltk.corpus.words.words())
# stopwords_ = set(stopwords.words('english'))
# punctuation_ = set(string.punctuation)
# stemmer = SnowballStemmer("english")
#
#
# def filter_tokens(sent):
#     return ([w for w in sent if not w in stopwords_ and not w in punctuation_])
#
#
# def text_preprocess(string):
#     word_list = []
#     sent_tokens = sent_tokenize(string)
#     tokens = [sent for sent in map(word_tokenize, sent_tokens)]
#
#     tokens_lower = [[word.lower() for word in sent]
#                     for sent in tokens]
#     tokens_filtered = list(map(filter_tokens, tokens_lower))
#     for tlist in tokens_filtered:
#         for w in tlist:
#             if w != '...':
#                 word_list.append(stemmer.stem(w))
#
#     return " ".join(word_list)
#
#
# def tfidf(corpus):
#     # Transform in Tf-idf
#     vectorizer_tfidf = TfidfVectorizer()
#
#     corpus_tfidf = vectorizer_tfidf.fit_transform(corpus)
#     return corpus_tfidf
#
#
# def cosin_df(df_reviews, df_tips, user_requirements):
#     df_reviews = df_reviews[['business_id', 'text']].groupby('business_id')['text'].apply(' '.join).reset_index()
#     df_tips = df_tips[['business_id', 'text']].groupby('business_id')['text'].apply(' '.join).reset_index()
#
#     # Join two dfs of reviews and tips based on business_id
#     df_review_tips = pd.merge(df_reviews, df_tips, on=["business_id"], how='outer')
#     df_review_tips['text'] = df_review_tips["text_x"] + " " + df_review_tips["text_y"]
#     df_review_tips = df_review_tips[['business_id', 'text']]
#
#     # print(review_tips.iloc[0]['text'])
#     df_review_tips['cleaned_text'] = df_review_tips['text'].apply(lambda t: text_preprocess(str(t)))
#     # print(review_tips_df.iloc[0]['cleaned_text'])
#
#     # Add user_requirements into reviews_tips_df
#     df_review_tips = df_review_tips.append({"business_id": "user_requirements",
#                                             "cleaned_text": user_requirements}, ignore_index=True)
#
#     return df_review_tips
#
#
# def cosin_recommender(df_business, df_reviews, df_tips, user_requirements):
#     df_review_tips = cosin_df(df_reviews, df_tips, user_requirements)
#     corpus = df_review_tips['cleaned_text']
#     # Text Transform
#     corpus_tfidf = tfidf(corpus)
#     doc_sim = cosine_similarity(corpus_tfidf, corpus_tfidf)
#
#     # print("Restaurant: {}".format(restaurantname))
#     # index = df_business[df_business['name'] == restaurantname].index
#     # indices = doc_sim[index][0].argsort()[::-1][:11]
#     # for ind in indices:
#     #     if ind != index:
#     #         print(df_business.iloc[ind]['name'])
#
#     # Add user_requirements into business_df
#     df_business = df_business.append({"business_id": "user_requirements"}, ignore_index=True)
#
#     business_index = df_business[df_business['business_id'] == "user_requirements"].index
#     indices = doc_sim[business_index][0].argsort()[::-1][:11]
#     for ind in indices:
#         if ind != business_index:
#             print(df_business.iloc[ind]['name'])
#
#
# def main():
#     # Load a dataset from json files
#     print('Loading dataset from json...')
#     # transfer json file to dataframe
#     df_reviews = pd.read_json("dataset/reviews.json", lines=True,
#                               dtype={'review_id': str, 'user_id': str, 'business_id': str, 'stars': int,
#                                      'date': str, 'text': str, 'useful': int, 'funny': int, 'cool': int})
#     # df_users = pd.read_json("dataset/users.json", lines=True,
#     #                        dtype={'user_id': str, 'name': str, 'review_count': int, 'useful': int, 'funny': int,
#     #                               'cool': int, 'average_stars': float})
#     df_business = pd.read_json("dataset/business.json", lines=True,
#                                dtype={'name': str, 'review_count': int, 'business_id': str, 'address': str, 'city': str,
#                                       'state': str, 'postal_code': str, 'latitude': float, 'longitude': float, 'stars': int,
#                                       'is_open': int, 'attributes': dict, 'categories': []})
#     df_tips = pd.read_json("dataset/tips.json", lines=True,
#                            dtype={'business_id': str, 'user_id': str, 'text': str})
#
#     cosin_recommender(df_business, df_reviews, df_tips, "I want to have dinner at the a beautiful view and inexpensive French restaurant. ")
#
#
# if __name__ == "__main__":
#     main()