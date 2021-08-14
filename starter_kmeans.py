import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import helper as hlp
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    X_expand = tf.expand_dims(X,0)
    MU_expand = tf.expand_dims(MU,1)
    distance = tf.reduce_sum(tf.square(X_expand-MU_expand),2)
    return tf.transpose(distance)

def training(k, n, d, data, valid_data, is_valid):
  np.random.seed(0)
  train_loss, val_loss = [], []
  X = tf.compat.v1.placeholder('float', shape=[None,d])
  MU = tf.Variable(tf.random.truncated_normal([k,d],stddev=0.05))
  distance = distanceFunc(X,MU)

  loss = tf.reduce_sum(tf.reduce_min(distance,axis=1))
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1,beta1=0.9,beta2=0.99,epsilon=1e-5)
  optimizer = optimizer.minimize(loss=loss)

  variables = tf.compat.v1.global_variables_initializer()
  sess = tf.compat.v1.InteractiveSession()
  sess.run(variables)

  for step in range(100):
    mu, loss_val, _ = sess.run([MU,loss,optimizer], feed_dict={X:data})
    train_loss.append(loss_val)
    if is_valid:
      val_mu, loss_val, _ = sess.run([MU,loss,optimizer], feed_dict={X:val_data})
      val_loss.append(loss_val)
  distance = distanceFunc(X,MU)
  clusters = tf.argmin(distance,1)
  clusters_assignment = sess.run(clusters, feed_dict={X:data, MU:mu})
  plt.plot(train_loss,label='Training')
  plt.plot(val_loss,label='Validation')
  plt.legend()
  plt.suptitle('K-Means Loss versus Iterations', fontsize=20)
  plt.title('Validation Loss: ' + str(loss_val), fontsize=10)
  plt.xlabel('Number of Iterations')
  plt.ylabel('Loss')
  plt.show()

  percentages = np.zeros((k))
  for point in clusters_assignment:
    percentages[point] += 1
  percentages = percentages / np.sum(percentages)

  sc = plt.scatter(data[:,0], data[:,1], c=clusters_assignment, cmap=plt.get_cmap('Set2'), s=25, alpha=0.6)
  lp = lambda i: plt.plot([],color=sc.cmap(sc.norm(i)), mec="none",
                        label="Cluster {:g}".format(i) + ": %.4f" %(percentages[i]), ls="", marker=".", markersize=20)[0]
  handles = [lp(i) for i in np.unique(clusters_assignment)]
  plt.legend(handles=handles)
  plt.scatter(mu[:,0], mu[:,1], marker='.', c='black', cmap=plt.get_cmap('Set1'), s=50, alpha = 0.8)
  plt.suptitle('Clustering', fontsize=20)
  plt.title('Validation Loss: ' + str(loss_val), fontsize=10)
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.show()

if __name__ == '__main__':
  # Loading data
  data = np.load('data2D.npy')
  # data = np.load('data100D.npy')
  [num_pts, dim] = np.shape(data)

  is_valid = True

  # For Validation set
  if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]
  
  # val_data = 0
  # training(3,num_pts,dim,data,val_data,is_valid)
  
  for k in range(1,6):
    training(k, num_pts, dim, data, val_data, is_valid)

  # K = [5,10,15,20,30]
  # for k in K:
  #     training(k,num_pts,dim,data,val_data,is_valid)