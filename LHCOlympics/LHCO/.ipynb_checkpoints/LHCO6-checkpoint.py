import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda, Layer
from tensorflow.keras.models import Sequential
import pandas as pd

f = pd.read_hdf("events_anomalydetection_DelphesPythia8_v2_qcd_features.h5")

G = (f[['pxj1','pxj2', 'pyj1', 'pyj2']]).to_numpy()
px1 = G[:, 0]
py1 = G[:, 2]
px2 = G[:, 1]
py2 = G[:, 3]

class MyLayer(Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._t1 = self.add_weight(name='x', 
                                    #shape=(1,),
                                    initializer=tf.keras.initializers.RandomUniform(minval= 0, maxval=2*np.pi), #'uniform',
                                    trainable=True)
        self._t2 = self.add_weight(name='x', 
                                    #shape=(1,),
                                    initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2*np.pi), #'uniform',
                                    trainable=True)
        self._t3 = self.add_weight(name='x', 
                                    #shape=(1,),
                                    initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2*np.pi), #'uniform',
                                    trainable=True)
        self._t4 = self.add_weight(name='x', 
                                    #shape=(1,),
                                    initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2*np.pi), #'uniform',
                                    trainable=True)
        self._t5 = self.add_weight(name='x', 
                                    #shape=(1,),
                                    initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2*np.pi), #'uniform',
                                    trainable=True)
        self._t6 = self.add_weight(name='x', 
                                    #shape=(1,),
                                    initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2*np.pi), #'uniform',
                                    trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, X):
        s1 = tf.math.sin(self._t1)
        c1 = tf.math.cos(self._t1)
        s2 = tf.math.sin(self._t2)
        c2 = tf.math.cos(self._t2)
        s3 = tf.math.sin(self._t3)
        c3 = tf.math.cos(self._t3)
        s4 = tf.math.sin(self._t4)
        c4 = tf.math.cos(self._t4)
        s5 = tf.math.sin(self._t5)
        c5 = tf.math.cos(self._t5)
        s6 = tf.math.sin(self._t6)
        c6 = tf.math.cos(self._t6)
        R1 = [[c1, s1, 0.0, 0.0], [-1.0*s1, c1, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] #px1 -> py1     
        R2 = [[c2, 0.0, s2, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0*s2, 0.0, c2, 0.0], [0.0, 0.0, 0.0, 1.0]] #px1 -> px2
        R3 = [[c3, 0.0, 0.0, s3], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-1.0*s3, 0.0, 0.0, c3]] #px1 -> py2
        R4 = [[1.0, 0.0, 0.0, 0.0], [0.0, c4, s4, 0.0], [0.0, -1.0*s4, c4, 0.0], [0.0, 0.0, 0.0, 1.0]] #px2 -> py1
        R5 = [[1.0, 0.0, 0.0, 0.0], [0.0, c5, 0.0, s5], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0*s5, 0.0, c5]] #px2 -> py1
        R6 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, c6, s6], [0.0, 0.0, -1.0*s6, c6]] #py1 -> py2
        S = tf.linalg.matmul(R1, tf.linalg.matmul(R2, (tf.linalg.matmul(R3, tf.linalg.matmul(R4, tf.linalg.matmul(R5, R6))))))
        #S = R1 @ R2 @ R3 @ R4 @ R5 @ R6
        M = tf.convert_to_tensor(S)
        M = tf.reshape(M, [4, 4])
        return tf.linalg.matmul(X, M)
    
def define_discriminator(n_inputs=4):
	model = Sequential()
	model.add(Dense(25, activation='relu', input_dim=n_inputs))
	model.add(Dense(25, activation='relu', input_dim=n_inputs))    
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(n_outputs=1):
	#model = Sequential()
	#model.add(Dense(15, activation='relu', input_dim=n_outputs))
	#model.add(Dense(15, activation='relu', input_dim=n_outputs))    
	#model.add(Dense(n_outputs, activation='linear'))

	mymodel_inputtest = Input(shape=(4,))
	mymodel_test = MyLayer()(mymodel_inputtest)
	model = Model(mymodel_inputtest, mymodel_test)
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
 
# generate n real samples with class labels
def generate_real_samples(n):
	randomlySelectedY = np.argsort(np.random.random(len(px1)))[:n]
	X = tf.convert_to_tensor([px1[randomlySelectedY], py1[randomlySelectedY], px2[randomlySelectedY], py2[randomlySelectedY]])
	y = np.ones((n, 1))
	return np.transpose(X), y
 
# generate points in latent space as input for the generator
def generate_latent_points(n):
	# generate points in the latent space
	x_input = generate_real_samples(n)
	return x_input[0]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, n):
	# generate points in latent space
	x_input = generate_latent_points(n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = np.zeros((n, 1))
	return X, y

def generate_fake_samples_with_input(generator, n):
	# generate points in latent space
	x_input = generate_latent_points(n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = np.zeros((n, 1))
	return X, y, x_input

k = 800
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, n_epochs=5*k, n_batch=128, n_eval=k):
	# determine half the size of one batch, for updating the discriminator
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# prepare real samples
		x_real, y_real = generate_real_samples(half_batch)
		# prepare fake examples
		x_fake, y_fake = generate_fake_samples(g_model, half_batch)
		# update discriminator
		d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)
		# prepare points in latent space as input for the generator
		x_gan = generate_latent_points(n_batch)
		# create inverted labels for the fake samples
		y_gan = np.ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)
		if (i+1) % n_eval == 0:
			print("epoch = ", i)
            
            
N = 12
t1i = []
t1f = []
t2i = []
t2f = []
t3i = []
t3f = []
t4i = []
t4f = []
t5i = []
t5f = []
t6i = []
t6f = []
for j in range(N):
    print("j = ", j)
    # create the discriminator
    discriminator = define_discriminator()
    # create the generator
    generator = define_generator()
    # create the gan
    gan_model = define_gan(generator, discriminator)
    t1i.append(generator.layers[-1].get_weights()[0])
    t2i.append(generator.layers[-1].get_weights()[1])
    t3i.append(generator.layers[-1].get_weights()[2])
    t4i.append(generator.layers[-1].get_weights()[3])
    t5i.append(generator.layers[-1].get_weights()[4])
    t6i.append(generator.layers[-1].get_weights()[5])

    # train model
    train(generator, discriminator, gan_model)
    t1f.append(generator.layers[-1].get_weights()[0])
    t2f.append(generator.layers[-1].get_weights()[1])
    t3f.append(generator.layers[-1].get_weights()[2])
    t4f.append(generator.layers[-1].get_weights()[3])
    t5f.append(generator.layers[-1].get_weights()[4])
    t6f.append(generator.layers[-1].get_weights()[5])
    print("t1i = ", t1i)
    print("t2i = ", t2i)
    print("t3i = ", t3i)
    print("t4i = ", t4i)
    print("t5i = ", t5i)
    print("t6i = ", t6i)
    print()
    print("t1f = ", t1f)
    print("t2f = ", t2f)
    print("t3f = ", t3f)
    print("t4f = ", t4f)
    print("t5f = ", t5f)
    print("t6f = ", t6f)

