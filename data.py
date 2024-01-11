import json as j
import pandas as pd
import re
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


json_data = None
with open('/Users/akselituominen/python/reviewStars/data/yelp_academic_dataset_review.json') as f:
    json_raw = f.readlines()
    joined_lines = "[" + ",".join(json_raw) + "]" # add square brackets to the outside
    json_data = j.loads(joined_lines) # convert to json object

data = pd.DataFrame(json_data) # convert to dataframe
stemmer = SnowballStemmer('english') #Instantiate Stemmer
#Basically used for reducing words back to their root form
words  = stopwords.words("english") #Instantiate Stopwords
#stopwords are super common words that dont really matter much like "and", "or"

#What we are doing here is removing all the stopwords, nusm and stemming the words
#Take from column [text] and put to [cleaned]
data['cleaned'] = data['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

#Splitting the data into training and testing, data.starts the target vector
X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.stars, test_size=0.2)


pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                        ('chi',  SelectKBest(chi2, k=10000)),
                        ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])
#ngram_range is the range of words to look at, so (1,2) looks at 1 word and 2 word phrases
#The advantage of this is that it looks at phrases like "not good" and "very good"

#Vectorizer creates a matrix of words and their frequencies
# Term Frequency (TF): This summarizes how often a given word appears within a document.
# Inverse Document Frequency (IDF): This reduces the weight of words that appear a lot across documents. It's a measure of how much information the word provides.

#Train on best k=10000 phrases
#As classifier we use standard vector classifier
#We use pipeline to chain vectorizer, chi and classifier

model = pipeline.fit(X_train, y_train)

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

feature_names = vectorizer.get_feature_names_out()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

target_names = ['1', '2', '3', '4', '5']
print("top 10 keywords per class:")
for i, label in enumerate(target_names):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (label, " ".join(feature_names[top10])))

print("accuracy score: " + str(model.score(X_test, y_test)))

#Put your own text here to test
print(model.predict(['Not your traditional kebab house. This place is just brilliant, They do really good gourmet kibabas restaurant style as well as falafel plates and rolls']))