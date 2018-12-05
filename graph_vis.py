from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random

import numpy as np
import scipy.spatial as spatial
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import sys

import plotly as ply
import plotly.graph_objs as go

import igraph as ig

#KELL:
#a graf a teljes szoveg legyen
#gyakran elofordulo szavak nagyobb/mas szinu csomopontokkent
#szavak koordinatait a word2vec adja
#word2vec altal visszaadott listaban nem szereplo szavak nem szerepelnek a grafban

#a program inditasi argumentumkent atvesz egy txt fajlt a feldolgozando szoveggel
#az argumentumkent kapott fajl utvonala
filename = sys.argv[1]

#fajl beolvasasa szolistaba
def read_data(filename):
    read_text = []
    #ezek a szavak eltavolitasra kerulnek mert nem relevansak
    bad_words = ['though', 'when', 'while', 'unless', 'until', 'for,', 'and', 'but', 'so', 'nor', 'yet', 'or', 'the', 'a', 'an', 'as', 'of', 'from', 'to', '(', ')', '.', '!', ',', ';', '?', ':', '-', '&', '/', '=', '-', '+']
    with open(filename, "r") as f:
        for line in f:
            for word in line.split():
                if word not in bad_words:
                    word = word.translate(None, '-,.;:()?!')
                    read_text.append(word)
    return read_text

#minden szo elfordulasanak megszamlalasa
def word_count(text):
    counts = dict()
    for word in text:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

# Read the data into a list of strings.
#def read_data(filename):
#  
#  with open(filename, 'r') as f:
#    data = tf.compat.as_str(f.read().split())
#  return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))

#print(word_count(vocabulary))

#Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  data_text = list()
  unk_count = 0
  for word in words:
    #if word in dictionary:
     # index = dictionary[word]
   # else:
    #  index = 0  # dictionary['UNK']
     # unk_count += 1
    index = dictionary.get(word, 0)
    if index == 0: #dictionary['UNK']
         unk_count += 1
    data.append(index)
    data_text.append(word)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data_text, data, count, dictionary, reversed_dictionary

data_text, data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
#print(data_text)

del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
#A SORT A DINAMIKUSSAG ERDEKEBEN ADTAM HOZZA
#HA A vocabulary_size EGYENLO A reverse_dictionary MERETEVEL, AKKOR MINDIG A SZOTAR VALOS MERETEVEL DOLGOZIK A PROGRAM
#NINCS TOBB KEYERROR
vocabulary_size = len(reverse_dictionary)

data_index = 0

# Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
      #buffer[:] = data[:span]
      #EZ CSAK FOR CIKLUSRA ATIRVA VOLT HAJLANDO NORMALISAN MUKODNI
      for word in data[:span]:
          buffer.append(word)
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
print(batch)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
#num_steps = 100001
num_steps = 10001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()


# dimenzio redukcio 3D-be
try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  
  tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=2500, method='barnes_hut')
  #MINDIG PONTOSAN ANNYI SZOT RAJZOL FEL AMENNYI A SZOTARBAN IS SZEREPEL
  plot_only = len(reverse_dictionary)
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  
except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')

#graf kirajzolasa

coordinates = [] #szavak koordinatait tarolja

#python listaba szedi a koordinata-parokat
for i, label in enumerate(labels):
    x, y, z = low_dim_embs[i, :]
    #print(x)
    #print(y)
    coordinates.append((x,y,z))
    
print(coordinates)

#a szohalot megvalosito graf peldanyositasa
wordGraph = ig.Graph()

x, y, z = [], [], []

for idx, label in enumerate(labels):
    #print(idx)
    #print(label)
    wordGraph.add_vertex()
    vertex = wordGraph.vs[wordGraph.vcount()-1]
    vertex["name"] = vertex.index
    vertex["label"] = label
    vertex["x"] = coordinates[idx][0]
    vertex["y"] = coordinates[idx][1]
    vertex["z"] = coordinates[idx][2]
    x.append(coordinates[idx][0])
    y.append(coordinates[idx][1])
    z.append(coordinates[idx][2])

x = np.array(x)
y = np.array(y)
z = np.array(z)

tree = spatial.KDTree(zip(x.ravel(), y.ravel(), z.ravel()))

edges = []

#for v in wordGraph.vs():
#    pts = np.array([[v["x"], v["y"], v["z"]]])
#    #list_nn = tree.query(pts, 10);
#    list_nn = tree.query(pts, 10)
#    print(list_nn)
#    for idx, nn in enumerate(list_nn[1][0]):
#        edges += [(v.index, nn)]

#data_text = read_data(filename)
        
#def search_neighbor(data_text, tword, n):
    
for v in wordGraph.vs():
    indices = [i for i, x in enumerate(data_text) if x == v["label"]]
    
    #print('indices for ' + v["label"])
    #print(indices)
    
    for k in indices:
        print(k)
        try:
            lbl = data_text[k-1]
            for h in wordGraph.vs():
                if h["label"] == lbl:
                    edges += [(v.index, h.index)]
                    print('Added edges between: ' + v["label"] + ' and ' + h["label"]) 
                #else:
                 #   print('Not found')
        except IndexError:
            print('Not found')
            
        try:
            lbl = data_text[k+1]
            for h in wordGraph.vs():
                if h["label"] == lbl:
                    edges += [(v.index, h.index)]
                    print('Added edges between: ' + v["label"] + ' and ' + h["label"]) 
                #else:
                 #   print('Not found')
        except IndexError:
            print('Not found')
    
wordGraph.add_edges(edges)

#tobbszoros elek es hurkok eltavolitasa
wordGraph=wordGraph.simplify()

#layout letrehozasa - a vertexek pont ott vannak ahova a koordinataik mutatnak

#print(wordGraph.get_edgelist())

N=len(labels)

Xe, Ye, Ze, Xn, Yn, Zn = [], [], [], [], [], []

glabel=list(wordGraph.vs["label"])

print(glabel)

for c in coordinates:
    Xn.append(c[0])
    Yn.append(c[1])
    Zn.append(c[2])
    
#print(Xn)
#print(Yn)
#print(Zn)

x_lines = list()
y_lines = list()
z_lines = list()

#create the coordinate list for the lines
for e in edges:
    x_lines.append(Xn[e[0]])
    y_lines.append(Yn[e[0]])
    z_lines.append(Zn[e[0]])
    x_lines.append(Xn[e[1]])
    y_lines.append(Yn[e[1]])
    z_lines.append(Zn[e[1]])
    
    x_lines.append(None)
    y_lines.append(None)
    z_lines.append(None)
    
#print(x_lines)
#print(y_lines)
#print(z_lines)
    
#ez a trace rajzolja ki az eleket
trace1=go.Scatter3d(x=x_lines,
               y=y_lines,
               z=z_lines,
               mode='lines',
               line= dict(color='rgb(210,210,210)', width=2),
               hoverinfo='none'
               )

#ez a trace rajzolja ki a vertexeket
trace2=go.Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               marker=dict(symbol='circle',
                                        size=5,
                                        color='#6959CD',
                                        line=dict(color='rgb(50,50,50)', width=0.5)
                                        ),
               text = glabel,
               hoverinfo='text'
               )

width = 1000
height = 1000

layout=go.Layout(  
    font= dict(size=12),
    showlegend=False,
    autosize=False,
    width=width,
    height=height,
    hovermode='closest',        
    )

data = [trace1, trace2]

fig = go.Figure(data = data, layout = layout)

ply.offline.plot({"data": data, "layout": layout}, auto_open = True, filename='output.html')
