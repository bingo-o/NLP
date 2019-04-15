import pandas as pd

df = pd.read_csv("../datasets/fasttext/labeledTrainData.tsv", sep="\t", encoding="utf-8")
X = df["review"].values
y = df["sentiment"].values

from bs4 import BeautifulSoup
import re
X_processed = []
for line in X:
	text = BeautifulSoup(line, "lxml").get_text()
	text = re.sub(r"[^a-zA-Z]", " ", text).lower()
	text = " ".join(text.split())
	X_processed.append(text)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y)

with open("../datasets/fasttext/train.txt", "w", encoding="utf-8") as f:
    for text, label in zip(X_train, y_train):
        f.write("__label__" + str(label) + "\t" + text + "\n")

with open("../datasets/fasttext/test.txt", "w", encoding="utf-8") as f:
    for text, label in zip(X_test, y_test):
        f.write("__label__" + str(label) + "\t" + text + "\n")


import fasttext
classifier = fasttext.supervised("../datasets/fasttext/train.txt", "../datasets/fasttext/model_clf")
# 加载模型
# classifier = fasttext.load_model("../datasets/fasttext/model_clf.bin")
result = classifier.test("../datasets/fasttext/test.txt")
print("precision:", result.precision, "recall:", result.recall)

texts = ["i was particularly moved by the understated courage and integrity of l anglaise in this beautifully acted intellectually and visually compelling film thank you so much monsieur le directeur rohmer", "i have to say i quite enjoyed soldier russell was very good as this trained psychopath rediscovering his humanity very watchable and nowhere near as bad as i d been led to believe yes it has problems but provides its share of entertainment"]
labels = classifier.predict(texts)
print(labels)


with open("../datasets/fasttext/train_vec.txt", "w", encoding="utf-8") as f:
    for text in X_train:
        f.write(text + "\n")

model_cbow = fasttext.cbow("../datasets/fasttext/train_vec.txt", "../datasets/fasttext/model_cbow")
print("cbow:", model_cbow["king"])

model_skipgram = fasttext.skipgram("../datasets/fasttext/train_vec.txt", "../datasets/fasttext/model_skipgram")
print("skipgram:", model_skipgram["king"])