import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import string
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

def conver_json_to_csv():
    df = pd.read_json('D:/2020 Summer/EBC5125 Data Science/Project/project_dataset/yelp_dataset/new_reviews.json',
                     lines=True,dtype={'review_id':str,'user_id':str,
                                 'business_id':str,'stars':int,
                                 'date':str,'text':str,'useful':int,
                                 'funny':int,'cool':int})
    df.to_csv (r'D:/2020 Summer/EBC5125 Data Science/Project/project_dataset/yelp_dataset/yelp_review_toronto.csv', index = None, header=True)

    df = pd.read_json('D:/2020 Summer/EBC5125 Data Science/Project/project_dataset/yelp_dataset/new_business.json',
                     lines=True)
    df.to_csv (r'D:/2020 Summer/EBC5125 Data Science/Project/project_dataset/yelp_dataset/yelp_business.csv', index = None, header=True)


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    stop = []
    for word in stopwords.words('english'):
        s = [char for char in word if char not in string.punctuation]
        stop.append(''.join(s))
    # Now just remove any stopwords
    return " ".join([word for word in nopunc.split() if word.lower() not in stop])


def matrix_factorization(R, P, Q, steps=25, gamma=0.001, lamda=0.02):
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    eij = R.loc[i, j] - np.dot(P.loc[i], Q.loc[j])
                    P.loc[i] = P.loc[i] + gamma * (eij * Q.loc[j] - lamda * P.loc[i])
                    Q.loc[j] = Q.loc[j] + gamma * (eij * P.loc[i] - lamda * Q.loc[j])
        e = 0
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    e = e + pow(R.loc[i, j] - np.dot(P.loc[i], Q.loc[j]), 2) + lamda * (
                                pow(np.linalg.norm(P.loc[i]), 2) + pow(np.linalg.norm(Q.loc[j]), 2))
        if e < 0.001:
            break

    return P, Q

def main():
    print("Convert json to csv")
    conver_json_to_csv()

    df = pd.read_csv('D:/2020 Summer/EBC5125 Data Science/Project/project_dataset/yelp_dataset/yelp_review_toronto.csv')
    df = df[['review_id', 'user_id', 'business_id', 'text', 'stars', 'date']]
    #df = df.loc[df['stars'] >= 3]
    #df = df.sample(n=10000)

    df_business = pd.read_csv('D:/2020 Summer/EBC5125 Data Science/Project/project_dataset/yelp_dataset/yelp_business.csv')

    # Select only stars and text
    yelp_data = df[['business_id', 'user_id', 'stars', 'text']]

    yelp_data['text'] = yelp_data['text'].apply(text_process)

    userid_df = yelp_data[['user_id', 'text']]
    business_df = yelp_data[['business_id', 'text']]
    userid_df = userid_df.groupby('user_id').agg({'text': ' '.join})
    business_df = business_df.groupby('business_id').agg({'text': ' '.join})

    from sklearn.feature_extraction.text import TfidfVectorizer
    # userid vectorizer
    userid_vectorizer = TfidfVectorizer(tokenizer=WordPunctTokenizer().tokenize, max_features=5000)
    userid_vectors = userid_vectorizer.fit_transform(userid_df['text'])
    print(userid_vectors.shape)

    # Business id vectorizer
    businessid_vectorizer = TfidfVectorizer(tokenizer=WordPunctTokenizer().tokenize, max_features=5000)
    businessid_vectors = businessid_vectorizer.fit_transform(business_df['text'])
    print(businessid_vectors.shape)

    userid_rating_matrix = pd.pivot_table(yelp_data, values='stars', index=['user_id'], columns=['business_id'])
    print(userid_rating_matrix.shape)

    P = pd.DataFrame(userid_vectors.toarray(), index=userid_df.index, columns=userid_vectorizer.get_feature_names())
    Q = pd.DataFrame(businessid_vectors.toarray(), index=business_df.index,
                     columns=businessid_vectorizer.get_feature_names())

    import time
    start_time = time.time()
    P, Q = matrix_factorization(userid_rating_matrix, P, Q, steps=25, gamma=0.001, lamda=0.02)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Store P, Q and vectorizer in pickle file
    import pickle
    output = open('yelp_recommendation_model.pkl', 'wb')
    pickle.dump(P, output)
    pickle.dump(Q, output)
    pickle.dump(userid_vectorizer, output)
    output.close()

    words = "i want to have some good pizza"
    test_df = pd.DataFrame([words], columns=['text'])
    test_df['text'] = test_df['text'].apply(text_process)
    test_vectors = userid_vectorizer.transform(test_df['text'])
    test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())

    predictItemRating = pd.DataFrame(np.dot(test_v_df.loc[0], Q.T), index=Q.index, columns=['Rating'])
    topRecommendations = pd.DataFrame.sort_values(predictItemRating, ['Rating'], ascending=[0])[:7]

    for i in topRecommendations.index:
        print(df_business[df_business['business_id'] == i]['name'].iloc[0])
        print(df_business[df_business['business_id'] == i]['categories'].iloc[0])
        print(str(df_business[df_business['business_id'] == i]['stars'].iloc[0]) + ' ' + str(
            df_business[df_business['business_id'] == i]['review_count'].iloc[0]))
        print('')

if __name__ == '__main__':
    main()


