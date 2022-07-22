import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import wordcloud
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

##### Loading Data set #####
MainData = pd.read_csv('articles1.csv')
pd.set_option('max_columns', None)

print("Loading data done.")
print("_________________________________________________")
####################################

##### Preprocessing Data set #####
MainData.rename(columns={'Unnamed: 0': 'Name'}, inplace=True)
MainData = MainData.drop(columns=['Name', 'id', 'publication', 'date', 'year', 'month', 'url'], axis=1)


def remove_punc(data):
    punc = '''!()-[]{};:'",<>./?@#$%^&*_~'''
    for ele in data:
        if ele in punc:
            data = data.replace(ele, "")
    return data


def convert_lower(data):
    for column in data.columns:
        data[column] = data[column].str.lower()
    return data


stop_words = stopwords.words("english")


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# nlp = spacy.load('en', disable=['parser', 'ner'])
MainData = remove_punc(MainData)
MainData = convert_lower(MainData)

print("Preprocessing done.")
print("_________________________________________________")
#####################################

##### Data Visualization #####
long_string = " ".join(MainData.title)
wordcloud = wordcloud.WordCloud()
wordcloud.generate(long_string)
wordcloud.to_image().show()


def plot_10_most_common_words(data_count, count_vectorizer_):
    words = count_vectorizer_.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in data_count:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    plt.bar(x_pos, counts, align='center')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title('10 most common words')
    plt.show()


CountVec = CountVectorizer(stop_words=stop_words)
Data = CountVec.fit_transform(MainData.title)
plot_10_most_common_words(Data, CountVec)
print("Visualization done.")
print("_________________________________________________")
#####################################

##### Model Training #####
Train_Data, Test_Data = train_test_split(Data, test_size=0.20, random_state=0)

number_topics = 10
number_words = 10
LDA = LatentDirichletAllocation(n_components=number_topics, random_state=0)
LDA.fit(Train_Data)

print("Model Training done.")
print("_________________________________________________")
##### Model Evaluation and Results #####


def print_topics(lda_model, feature_names, n_top_words):
    words = feature_names.get_feature_names()

    topic_words = lda_model.components_
    for topic_idx, topic in enumerate(topic_words):

        print("\nMost common words in topic %d:" % (topic_idx + 1))
        # sort top words according to their value
        print(", ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


print("Topics found via LDA:")
print_topics(LDA, CountVec, number_words)
print("_________________________________________________")

Train_Prep = LDA.perplexity(Train_Data)
Train_score = LDA.score(Train_Data)
print("Train perplexity: ", Train_Prep)
print("Train score: ", Train_score)

Test_Prep = LDA.perplexity(Test_Data)
Test_score = LDA.score(Test_Data)
print("Test perplexity: ", Test_Prep)
print("Test score: ", Test_score)
print("_________________________________________________")