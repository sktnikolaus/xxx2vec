
from __future__ import absolute_import
from __future__ import print_function


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
import pickle

vecs_demo = pickle.load(open( "save_demo.p", "rb" ))
vecs_repu = pickle.load(open( "save_repu.p", "rb" ))

vecs = pickle.load(open( "save.p", "rb" ))
reverse_dictionary = pickle.load(open('reverse.p','rb'))

def plot_with_labels(low_dim_embs,low_dim_embs2, labels, filename='tsne1.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
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
  plt.savefig(filename)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 800
p = 1436

vecs1 = vecs[:plot_only,:]
vecs2 = vecs[p:p+plot_only,:]

vecs_tot = np.concatenate((vecs1,vecs2),axis=0)
print(vecs_tot.shape)

'''
lows = tsne.fit_transform(vecs_tot)


low_dim_embs = lows[:plot_only,:]#tsne.fit_transform(vecs[:plot_only,:])
low_dim_embs2 = lows[plot_only:,:]#tsne.fit_transform(vecs[p:p+plot_only,:])
labels = [i for i in xrange(plot_only)]#[reverse_dictionary[i] for i in xrange(plot_only)]
#plot_with_labels(low_dim_embs, low_dim_embs2,labels)
'''
y1 = [0 for i in xrange(p)]
y2 = [1 for i in xrange(vecs.shape[0]-p)]

y = np.concatenate((y1,y2),axis=0)

print(y.shape,vecs.shape)

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

shuffled,labels = shuffle(vecs,y)

print(shuffled.shape)

print(labels[:20])

train_vecs = shuffled[:2000]
y_train = labels[:2000]
test_vecs = shuffled[2000:]
y_test = labels[2000:]

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))




