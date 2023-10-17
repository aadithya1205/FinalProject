import pandas as pd
data = pd.read_csv('twitter_new.csv',encoding="latin-1",header=0,names=["target","id","date","flag","user","text"],error_bad_lines=False, engine="python")
data.dropna(inplace=True)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
#getting rid of stop words
cv = CountVectorizer(stop_words='english',ngram_range = (1,3),tokenizer = token.tokenize)
text_counts = cv.fit_transform(data["text"])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(text_counts, data["target"], test_size=0.25, random_state=5)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
classifier= MultinomialNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
from sklearn import metrics
predicted = classifier.predict(x_test)
accuracy_score = metrics.accuracy_score(predicted, y_test)

print("Accuracy:",accuracy_score)
