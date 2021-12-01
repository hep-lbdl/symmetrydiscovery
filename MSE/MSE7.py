import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Dense, Input, Layer, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import scipy
from matplotlib import gridspec
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import mse, binary_crossentropy
from keras.layers.merge import concatenate, Concatenate
from keras.layers import concatenate
import tensorflow.keras.backend as K

class MyLayer(Layer):

    def __init__(self, kernel_initilizer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1), **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initilizer)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._c = self.add_weight(name='x',
                                    initializer=self.kernel_initializer, #'uniform',
                                    trainable=True)
        self._s = self.add_weight(name='x',
                                    initializer=self.kernel_initializer, #'uniform',
                                    trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, X):
        npc = [[self._c,-1.0*self._s],[self._s,self._c]]
        M = tf.convert_to_tensor(npc)
        M = tf.reshape(M, [2, 2])
        return tf.linalg.matmul(X, M)
    
def myloss2d(y_true, y_pred, alpha = 0.1):
    #alpha determines the amount of decorrelation; 0 means no decorrelation.
    
    #We want to learn g(g(x)) = x with g != identity and g(x) and x should have the same probability density.
    #x = y_pred[:,0]
    #g(g(x)) = y_pred[:,1]
    #h(g(x)) = y_pred[:,2]

    return binary_crossentropy(y_pred[:,4], y_true) + alpha*mse(y_pred[:,0:2],y_pred[:,2:4])

#Quick vanilla GAN from https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/

# define the standalone discriminator model
def define_discriminator(n_inputs=2):
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

	mymodel_inputtest = Input(shape=(2,))
	mymodel_test = MyLayer()(mymodel_inputtest)
	model = Model(mymodel_inputtest, mymodel_test)
	return model
 
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False

    myinput_2d = Input(shape=(2,))
    encoded_2d = generator(myinput_2d)
    myidentity = Lambda(lambda x: x)(myinput_2d)
    encoder_2d = Model(myinput_2d, encoded_2d)
    encoder2_2d = encoder_2d(encoder_2d(encoder_2d(encoder_2d(encoder_2d(encoder_2d(encoder_2d(myinput_2d)))))))
    autoencoder_2d = Model(myinput_2d, encoder2_2d)
    
    discriminated_2d = discriminator(myinput_2d)
    discriminator_2d = Model(myinput_2d, discriminated_2d)
    discriminator2_2d = discriminator_2d(encoder_2d(myinput_2d))
    model_discriminator2_2d = Model(myinput_2d, discriminator2_2d)
    
    justinput = Model(myinput_2d, myidentity)
    combinedModel_2d = Model(myinput_2d,Concatenate(axis=-1)([myidentity, encoder2_2d, discriminator2_2d]))
    
    # compile model
    combinedModel_2d.compile(loss=lambda y_true, y_pred: myloss2d(y_true, y_pred), optimizer='adam')
    return combinedModel_2d
 
# generate n real samples with class labels
def generate_real_samples(n):
	X = np.random.multivariate_normal([0, 0], [[1, 0],[0, 1]],n)
	y = ones((n, 1))
	return X, y
 
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
	y = zeros((n, 1))
	return X, y

def generate_fake_samples_with_input(generator, n):
	# generate points in latent space
	x_input = generate_latent_points(n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n, 1))
	return X, y, x_input


k = 2000
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
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)
		if (i+1) % n_eval == 0:
			print("epoch = ", i)
            
            
N = 50
c_i = []
s_i = []
c_f = []
s_f = []

for j in range(N):
    print("j = ", j)
    # create the discriminator
    discriminator = define_discriminator()
    # create the generator
    generator = define_generator()
    # create the gan
    gan_model = define_gan(generator, discriminator)
    c_i.append(generator.layers[-1].get_weights()[0])
    s_i.append(generator.layers[-1].get_weights()[1])
    # train model
    train(generator, discriminator, gan_model)
    c_f.append(generator.layers[-1].get_weights()[0])
    s_f.append(generator.layers[-1].get_weights()[1])
    print("c_i = ", c_i)
    print("s_i = ", s_i)
    print("c_f = ", c_f)
    print("s_f = ", s_f)