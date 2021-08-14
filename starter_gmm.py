import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import helper as hlp
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    X_expand = tf.expand_dims(X,0)
    MU_expand = tf.expand_dims(MU,1)
    distance = tf.reduce_sum(tf.square(X_expand-MU_expand),2)
    return tf.transpose(distance)

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    distance = distanceFunc(X,mu)
    exp = distance / (2*tf.squeeze(sigma))
    coef = -0.5*tf.to_float(tf.rank(X))*tf.log(2*np.pi*tf.squeeze(sigma))
    return coef - exp

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    log_probability = tf.squeeze(log_pi) + log_PDF
    log_sum = hlp.reduce_logsumexp(log_probability + tf.squeeze(log_pi), keep_dims=True)
    return log_probability - log_sum

def training(k, n, d, data, valid_data, is_valid):
    np.random.seed(0)
    train_loss, val_loss = [], []
    X = tf.compat.v1.placeholder('float', shape=[None,d])
    MU = tf.Variable(tf.random_normal([k,d],stddev=0.05))
    sigma = tf.exp(tf.Variable(tf.random_normal([k,1],stddev=0.05)))
    log_PDF = log_GaussPDF(X,MU,sigma)
    log_pi = tf.squeeze(hlp.logsoftmax(tf.Variable(tf.random_normal([k,1],stddev=0.05))))

    loss = -tf.reduce_sum(hlp.reduce_logsumexp(log_PDF+log_pi, 1))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1,beta1=0.9,beta2=0.99,epsilon=1e-5)
    optimizer = optimizer.minimize(loss=loss)

    predictions = tf.argmax(tf.nn.softmax(log_posterior(log_PDF, log_pi)),1)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        for step in range(100):
            mu, loss_val, _, assign = sess.run([MU,loss,optimizer,predictions], feed_dict={X:data})
            train_loss.append(loss_val)
            if is_valid:
                _, loss_val, _, _ = sess.run([MU,loss,optimizer,predictions], feed_dict={X:val_data})
                val_loss.append(loss_val)
        # print('log pi values: ' + str(log_pi.eval()))
        # print('sigma value: ' + str(sigma.eval()))
        # print('mu value: ' + str(MU.eval()))
    plt.plot(train_loss,label='Training')
    plt.plot(val_loss,label='Validation')
    plt.legend()
    plt.suptitle('K-Means Loss versus Iterations', fontsize=20)
    plt.title('Validation Loss: ' + str(loss_val), fontsize=10)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.show()

    percentages = np.zeros((k))
    for point in assign:
        percentages[point] += 1
    percentages = percentages / np.sum(percentages)

    sc = plt.scatter(data[:,0], data[:,1], c=assign, cmap=plt.get_cmap('Set2'), s=25, alpha=0.6)
    lp = lambda i: plt.plot([],color=sc.cmap(sc.norm(i)), mec="none",
                        label="Cluster {:g}".format(i) + ": %.4f" %(percentages[i]), ls="", marker=".", markersize=20)[0]
    handles = [lp(i) for i in np.unique(assign)]
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
  
#   val_data = 0
#   training(3,num_pts,dim,data,val_data,is_valid)

  for k in range(1,6):
    print(k)
    training(k, num_pts, dim, data, val_data, is_valid)

#   K = [5,10,15,20,30]
#   for k in K:
#       training(k,num_pts,dim,data,val_data,is_valid)
