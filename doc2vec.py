
# following this tutorial:
# https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis
# but not using it for sentiment analysis

import gensim
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import numpy as np
import random

LabeledSentence = gensim.models.doc2vec.LabeledSentence

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

subs_demo = read_data(filename_demo)
subs_repu = read_data(filename_repu)
subs = np.concatenate((subs_repu,subs_demo))

# create corresponding labels
y_demo = np.zeros(len(subs_demo))
y_repu = np.ones(len(subs_repu))

y = np.concatenate((y_demo,y_repu))

# We shuffle the da
#x,x_test,y_train,y_test = train_test_split(np.concatenate((subs_demo,subs_repu)),y,test_size=0.0)
shuffled,labels = shuffle(subs,y)

indices0 = [i for i,x in enumerate(labels) if x==0.]
indices1 = [i for i,x in enumerate(labels) if x==1.]

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

x = cleanText(shuffled)
#x_test = cleanText(x_test)

def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x = labelizeReviews(x, 'LAB')
#x_test = labelizeReviews(x, 'TEST')

print x[0]
size = 400

#instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
#model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

print 'initialized'
#build vocab over all reviews
model_dm.build_vocab(x)#np.concatenate((x,x_test)))
#model_dbow.build_vocab(x)
print 'built voc'
#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
all_train_reviews = x# np.array(x)#np.concatenate((x,x_test))
#print all_train_reviews.shape[0]
for epoch in range(3):
    print 'Epoch',epoch
    #perm = np.random.permutation(all_train_reviews.shape[0])#np.array(np.random.permutation(len(all_train_reviews)))#(all_train_reviews.shape[0])
    perm = range(len(all_train_reviews))
    shuffle(perm)
    perm_sentences = [all_train_reviews[i] for i in perm]
    #print all_train_reviews.shape[0]
    #sys.exit(-1)

    model_dm.train(perm_sentences)
    #model_dm.train(all_train_reviews[perm])
    #model_dbow.train(all_train_reviews[perm])

print 'train success'

#Get training set vectors from our models
def getVecs(model, corpus, size):
    #vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]

    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

train_vecs_dm = getVecs(model_dm, x, size)
#train_vecs_dbow = getVecs(model_dbow, x, size)

#train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

#print 'vec',train_vecs_dm[0]


'''
#train over test set
x_test = np.array(x_test)

for epoch in range(10):
    perm = np.random.permutation(x_test.shape[0])
    model_dm.train(x_test[perm])
    model_dbow.train(x_test[perm])

#Construct vectors for test reviews
test_vecs_dm = getVecs(model_dm, x_test, size)
test_vecs_dbow = getVecs(model_dbow, x_test, size)

test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
'''

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re

def shorten_text(text):  
  text = text.replace(',','')
  text = text.replace('"','')
  text = text.replace("'",'')

  #text_a = ' '.join(text_a)
  #text_a = ''.join(e for e in text_a if e.isalnum())
  text_a = re.sub('\W+',' ', text)
  text_a = text_a.split(' ')
  if len(text_a)>30:
    text_a = text_a[:15]+['...']+text_a[-10:]

  text_a = ' '.join(text_a)
  return '"'+text_a+'"'

def plot_with_labels(low_dim_embs,low_dim_embs2, labels, subs0,subs1,filename='tsne_doc7.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    if i==0:
      plt.scatter(x, y,color='blue',label='Democrats')
    else:
      plt.scatter(x, y,color='blue')
    '''
    plt.annotate(label,
                 xy=(x, y),
                 color='blue',
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    '''
  for i, label in enumerate(labels):
    x, y = low_dim_embs2[i,:]
    if i==0:
      plt.scatter(x, y,color='red',label='Republicans')
    else:
      plt.scatter(x, y,color='red')  
    '''
    plt.annotate(label,
                 xy=(x, y),
                 color='red',
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    '''

  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
  plt.savefig(filename)

  # write to output file
  filename = 'data1.txt'
  with open(filename,'w') as infile:

    infile.write('["democrats_x",')
    for j in xrange(len(labels)):
      if j==len(labels)-1:
        infile.write(str(low_dim_embs[j,0])+']\n')
      else:
        infile.write(str(low_dim_embs[j,0])+',')

    infile.write('["democrats",')
    for j in xrange(len(labels)):
      if j==len(labels)-1:
        infile.write(str(low_dim_embs[j,1])+']\n')
      else:
        infile.write(str(low_dim_embs[j,1])+',')

    infile.write('["republicans_x",')
    for j in xrange(len(labels)):
      if j==len(labels)-1:
        infile.write(str(low_dim_embs2[j,0])+']\n')
      else:
        infile.write(str(low_dim_embs2[j,0])+',')

    infile.write('["republicans",')
    for j in xrange(len(labels)):
      if j==len(labels)-1:
        infile.write(str(low_dim_embs2[j,1])+']\n')
      else:
        infile.write(str(low_dim_embs2[j,1])+',')

    infile.write('var labels_demo =[')
    for j in xrange(len(labels)):
      string = shorten_text(subs0[j])
      if j == len(labels)-1:
        infile.write(string+']\n')
      else:
        infile.write(string+',')

    infile.write('var labels_repu =[')
    for j in xrange(len(labels)):
      string = shorten_text(subs1[j])
      if j == len(labels)-1:
        infile.write(string+']\n')
      else:
        infile.write(string+',')

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 900

vecs0 = train_vecs_dm[indices0]
vecs1 = train_vecs_dm[indices1]

p = vecs0.shape[0]#1436

subs0 = shuffled[indices0]
subs1 = shuffled[indices1]

print vecs0.shape,vecs1.shape

#vecs0 = vecs0[:plot_only,:]
#vecs1 = vecs1[:plot_only,:]

vecs_tot = np.concatenate((vecs0,vecs1),axis=0)
print(vecs_tot.shape)

lows = tsne.fit_transform(vecs_tot)


low_dim_embs = lows[:plot_only,:]#tsne.fit_transform(vecs[:plot_only,:])
low_dim_embs2 = lows[p:p+plot_only,:]#tsne.fit_transform(vecs[p:p+plot_only,:])

subs0 = subs0[:plot_only]
subs1 = subs1[:plot_only]

labels = [i for i in xrange(plot_only)]#[reverse_dictionary[i] for i in xrange(plot_only)]
plot_with_labels(low_dim_embs, low_dim_embs2,labels,subs0,subs1)

