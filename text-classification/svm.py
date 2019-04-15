import pandas as pd

cnews = pd.read_csv("../datasets/cnews.train.txt", sep="\t", names=["category", "cnews"], encoding="utf-8").dropna()
X = cnews["cnews"].values
y = cnews["category"].values

import jieba
stopwords = pd.read_csv("../datasets/stopwords.txt", sep="\t", names=["stopword"], encoding="utf-8", quoting=3)
stopwords = stopwords["stopword"].values
X_processed = []
for line in X:
	segs = jieba.cut(line)
	segs = filter(lambda x: len(x) > 1, segs)
	segs = filter(lambda x: x not in stopwords, segs)
	X_processed.append(" ".join(segs))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y)

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

from sklearn.svm import SVC
svm_clf = SVC(kernel="linear")
svm_clf.fit(X_train, y_train)
print(svm_clf.score(X_test, y_test))