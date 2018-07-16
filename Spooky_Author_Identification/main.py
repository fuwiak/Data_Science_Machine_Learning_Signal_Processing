import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv("train.csv")


EAP_len = data[data['author'] == 'EAP'].shape[0]
HPL_len = data[data['author'] == 'HPL'].shape[0]
MWS_len = data[data['author'] == 'MWS'].shape[0]


# plt.bar(10,EAP_len,3, label="EAP")
# plt.bar(15,HPL_len,3, label="HPL")
# plt.bar(20,MWS_len,3, label="MWS")
# plt.legend()
# plt.ylabel('Number of examples')
# plt.title('Proportion of examples')
#plt.show()

def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


from nltk.corpus import stopwords
sw = stopwords.words('english')

def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)

data['text'] = data['text'].apply(stopwords)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
count_vectorizer = CountVectorizer()
count_vectorizer.fit(data['text'])
dictionary = count_vectorizer.vocabulary_.items()  

vocab = []
count = []
for key, value in dictionary:
    vocab.append(key)
    count.append(value)

vocab_bef_stem = pd.Series(count, index=vocab)
vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

def stemming(text):    
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

data['text'] = data['text'].apply(stemming)
tfid_vectorizer = TfidfVectorizer("english")
tfid_vectorizer.fit(data['text'])
dictionary = tfid_vectorizer.vocabulary_.items()

vocab = []
count = []

for key, value in dictionary:
    vocab.append(key)
    count.append(value)

vocab_after_stem = pd.Series(count, index=vocab)
vocab_after_stem = vocab_after_stem.sort_values(ascending=False)

top_vacab = vocab_after_stem.head(20)
#top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (15120, 15145))
#plt.show()

def length(text):
    return len(text)
data['length'] = data['text'].apply(length)

EAP_data = data[data['author'] == 'EAP']
HPL_data = data[data['author'] == 'HPL']
MWS_data = data[data['author'] == 'MWS']

import matplotlib

# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# bins = 500
# plt.hist(EAP_data['length'], alpha = 0.6, bins=bins, label='EAP')
# plt.hist(HPL_data['length'], alpha = 0.8, bins=bins, label='HPL')
# plt.hist(MWS_data['length'], alpha = 0.4, bins=bins, label='MWS')
# plt.xlabel('length')
# plt.ylabel('numbers')
# plt.legend(loc='upper right')
# plt.xlim(0,300)
# plt.grid()
# plt.show()

tfid_matrix = tfid_vectorizer.transform(data['text'])
array = tfid_matrix.todense()
df = pd.DataFrame(array)


#training model

df['output'] = data['author']
df['id'] = data['id']
output = 'output'
features.remove(output)
features.remove('id')

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV


alpha_list1 = np.linspace(0.006, 0.1, 20)
alpha_list1 = np.around(alpha_list1, decimals=4)
parameter_grid = [{"alpha":alpha_list1}]


#CL1
classifier1 = MultinomialNB()
gridsearch1 = GridSearchCV(classifier1,parameter_grid, scoring = 'neg_log_loss', cv = 4)
gridsearch1.fit(df[features], df[output])
results1 = pd.DataFrame()
results1['alpha'] = gridsearch1.cv_results_['param_alpha'].data
results1['neglogloss'] = gridsearch1.cv_results_['mean_test_score'].data

# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# plt.plot(results1['alpha'], -results1['neglogloss'])
# plt.xlabel('alpha')
# plt.ylabel('logloss')
# plt.grid()

# print("Best parameter: ",gridsearch1.best_params_)
# print("Best score: ",gridsearch1.best_score_) 


#CL2

alpha_list2 = np.linspace(0.006, 0.1, 20)
alpha_list2 = np.around(alpha_list2, decimals=4)
parameter_grid = [{"alpha":alpha_list2}]

classifier2 = MultinomialNB()
gridsearch2 = GridSearchCV(classifier2,parameter_grid, scoring = 'neg_log_loss', cv = 4)
gridsearch2.fit(df[features], df[output])

results2 = pd.DataFrame()
results2['alpha'] = gridsearch2.cv_results_['param_alpha'].data
results2['neglogloss'] = gridsearch2.cv_results_['mean_test_score'].data

# print("Best parameter: ",gridsearch2.best_params_)
# print("Best score: ",gridsearch2.best_score_) 