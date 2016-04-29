from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


with open('MaskedDataRaw.csv') as f:
    f.readline()
    for i, line in enumerate(f.readlines()):
        s = line.split(',')[3:]
        sentence.append(s)

pos =[]
neg = []
neu = []
for i in range(10):
    s = sentence[i][0]
    print(s)
    ss = sid.polarity_scores(s)
    print(ss)
    # for k in (ss):
    pos.append(ss['pos'])
    neg.append(ss['neg'])
    neu.append(ss['neu'])

pos_array = np.array(pos)
neg_array = np.array(neg)
neu_array = np.array(neu)
sen = np.column_stack((pos_array, neg_array, neu_array))
np.savetxt("sen.csv", sen)
X_new = np.column_stack((X, sen))
np.savez("X_new.npz", X = X_new)
np.savez("pos.npz", pos = pos)
np.savez("neg.npz", neg = neg)
np.savez("neu.npz", neu = neu)