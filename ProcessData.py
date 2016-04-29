import os, cPickle, re, sys
import numpy as np
import scipy.sparse
import math

vocab_size = 5000
data_path = "MaskedDataRaw.csv"
vocab_path = "temp.pkl"
save_path = "lol.npz"

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
    # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(
        r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

_DIGIT_RE = re.compile(r"\d")

def tokenize(s):
    """This function tokenize a sentence into a list of words/token"""
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    """This function tokenizes a sentence and keep emoticon"""
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for \
                   token in tokens]
    return tokens


def create_vocabulary(data_path, vocab_path, vocab_size, 
			normalize_digits, header):
    """Creating the vocabulary of words. It then keeps the most popular words,
    with size given by vocab_size
    It then save the vocabulary to disk"""
    print("Creating vocabulary at %s" % vocab_path)
    vocab = {}
    with open(data_path) as f:
        if header: f.readline()
        for i, line in enumerate(f.readlines()):
            if i % 10000 == 0: 
                sys.stdout.write(str(i) + ".")
                sys.stdout.flush()
            word_list = preprocess(','.join(line.split(',')[3:]), True)
            for word in word_list:
                # word = word.lower()
                if word.isdigit(): word = re.sub(_DIGIT_RE, "0", word) if \
                        normalize_digits else word 
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    vocab_list = vocab_list[:vocab_size]
    cPickle.dump(vocab_list, open(vocab_path, 'wb'))
    return vocab

def initialize_vocabulary(data_path, vocab_path, vocab_size,
        normalize_digits, header):
    """Load the vocabulary saved to disk, and create a dictionary and a reversed
    dictionary"""
    if not os.path.isfile(vocab_path): 
        create_vocabulary(data_path, vocab_path, vocab_size, 
                normalize_digits, header)
    rev_vocab = cPickle.load(open(vocab_path))
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab

def tweet_to_count_data(data_path, vocab_path, vocab_size,
                                            normalize_digits, header, pos, neg, neu):
    """Create a count matrix, for number of row equal to number of movies, and
    number of columns equal to number of words in the vocabulary. Each entry at
    row i column j denote how many time word j'th appear in tweets of movie i'th
    """
    vocab, rev_vocab = initialize_vocabulary(data_path, vocab_path,
            vocab_size, normalize_digits, header)

    n = 0
    with open(data_path) as f:
	for line in f.readlines(): n += 1
    if header: n -= 1
    p = len(vocab)+6
    X = scipy.sparse.lil_matrix((n, p), dtype = 'uint8')
    pos_count= np.zeros(shape=(p,1))
    neg_count= np.zeros(shape=(p,1))
    print("Creating the count matrix (%d, %d)" %(n, p))

    with open(data_path) as f:
        if header: f.readline()
        for i, line in enumerate(f.readlines()):
            if i % 1000 == 0:
                sys.stdout.write(str(i) + '.')
                sys.stdout.flush()
            word_list = preprocess(','.join(line.split(',')[3:]), True)
            y = line.split(',')[1]
            # print(y)
            for word in word_list:
                # word = word.lower()
                if word.isdigit(): word = re.sub(_DIGIT_RE, "0", word) if \
                        normalize_digits else word
                if word in vocab:
                    X[i, vocab[word]] += 1
                    if y == "1":
                        pos_count[vocab[word], 0] += 1
                        print(pos_count[vocab[word], 0])
                    elif y == "0":
                        neg_count[vocab[word], 0] +=1
            X[i, vocab_size] = len(','.join(line.split(',')[3:]).split())
            X[i, vocab_size+1] = pos[i]
            X[i, vocab_size+2] = neg[i]
            X[i, vocab_size+3] = neu[i]
            # for j in range(2000):
            #     X[i,j] = math.log1p(X[i, j]+1)
        for i in range(1528627):
            pos_word = 0
            neg_word = 0
            for j in range(len(vocab)):
                if pos_count[j] > neg_count[j]:
                    pos_word += X[i , j]
                elif pos_count[j] < neg_count[j]:
                    neg_word += X[i , j]
            X[i, vocab_size + 4] = pos_word
            X[i, vocab_size + 5] = neg_word
    return X, pos_count, neg_count

def main():
    if not os.path.exists(save_path):
        pos1 = np.load("pos.npz")['pos']
        neg1 = np.load("neg.npz")['neg']
        neu1 = np.load("neu.npz")['neu']
        X, pos, neg = tweet_to_count_data(data_path, vocab_path, 
                vocab_size, True, True, pos1, neg1, neu1)
        X = X.tocsc()
        with open(data_path) as f:
            f.readline()
            y = [int(line.split(',')[1]) for line in f.readlines()]
            y = np.array(y)
        np.savez(save_path, X = X, y = y, pos = pos, neg = neg)



if __name__ == "__main__":
    main()