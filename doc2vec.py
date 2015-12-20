
# following this tutorial:
# https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis
# but not using it for sentiment analysis

import gensim

LabeledSentence = gensim.models.doc2vec.LabeledSentence

from sklearn.cross_validation import train_test_split
import numpy as np

filename_demo = 'topic_sql_democrats_clean.txt'

filename_repu = 'topic_sql_republican_clean.txt'

def read_data(filename):
  ''' returns posts and words in correct format'''
  subs = []
  words = []
  with open(filename,'r') as infile:
    for line in infile:
      if line[0]!='\n':
        l = line#line.split(' ')
        subs.append(l)
        #for word in l:
        #  words.append(word)
  return subs#,words

subs_repu = read_data(filename_repu)
subs_demo = read_data(filename_demo)

# create corresponding labels
y_repu = np.zeros(len(subs_repu))
y_demo = np.ones(len(subs_demo))

y = np.concatenate((y_repu,y_demo))

x_train,x_test,y_train,y_test = train_test_split(np.concatenate((subs_repu,subs_demo)),y,test_size=0.2)

#Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)

def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')
unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')






