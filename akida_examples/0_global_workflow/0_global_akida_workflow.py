
#%%
###################### 1 - Create and train ######################

#%% [1.1 - Load and reshape MNIST dataset]
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tf_keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Add a channels dimension to the image sets as Akida expects 4-D inputs (corresponding to
# (num_samples, width, height, channels). Note: MNIST is a grayscale dataset and is unusual
# in this respect - most image data already includes a channel dimension, and this step will
# not be necessary.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Display a few images from the test set
f, axarr = plt.subplots(1, 4)
for i in range(0, 4):
    axarr[i].imshow(x_test[i].reshape((28, 28)), cmap=cm.Greys_r)
    axarr[i].set_title('Class %d' % y_test[i])
plt.show()

#%% [1.2 - Model definition]

import tf_keras as keras

model_keras = keras.models.Sequential([
    keras.layers.Rescaling(1. / 255, input_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=32, kernel_size=3, strides=2),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    # Separable layer
    keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', strides=2),
    keras.layers.Conv2D(filters=64, kernel_size=1, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
], 'mnistnet')

model_keras.summary()

#%% [1.3 - Model training]
from tf_keras.optimizers import Adam

model_keras.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy'])

_ = model_keras.fit(x_train, y_train, epochs=10, validation_split=0.1)

#%% [1.4 - Model Evaluation]
score = model_keras.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])

#%%
###################### 2 - Quantize ######################
# %% [2.1. - 8-bit quantization]
from quantizeml.models import quantize, QuantizationParams

qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
model_quantized = quantize(model_keras, qparams=qparams)

# %% [2.1. - 8-bit quantization summary]
model_quantized.summary()

# %% [2.1. - 8-bit quantization model evaluation]
def compile_evaluate(model):
    """ Compiles and evaluates the model, then return accuracy score. """
    model.compile(metrics=['accuracy'])
    return model.evaluate(x_test, y_test, verbose=0)[1]


print('Test accuracy after 8-bit quantization:', compile_evaluate(model_quantized))

# %% [2.2. - Effect of calibration]
model_quantized = quantize(model_keras, qparams=qparams,
                           samples=x_train, num_samples=1024, batch_size=100, epochs=2)
# %% [2.2. - Effect of calibration model evaluation]
print('Test accuracy after calibration:', compile_evaluate(model_quantized))

# %% [2.3. - 4-bit quantization]
qparams = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)
model_quantized = quantize(model_keras, qparams=qparams,
                           samples=x_train, num_samples=1024, batch_size=100, epochs=2)
# %% [2.3. - 4-bit quantization model evaluation]
print('Test accuracy after 4-bit quantization:', compile_evaluate(model_quantized))

# %% [2.4. - Model fine tuning (Quantization Aware Training)]
model_quantized.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy'])

model_quantized.fit(x_train, y_train, epochs=5, validation_split=0.1)

# %% [2.4. - Model fine tuning (Quantization Aware Training) model evaluation]
score = model_quantized.evaluate(x_test, y_test, verbose=0)[1]
print('Test accuracy after fine tuning:', score)

#%%
###################### 3 - Convert ######################
# %% [3.1. - Convert to Akida model]
from cnn2snn import convert

model_akida = convert(model_quantized)
model_akida.summary()

# %% [3.2. - Check performance]
accuracy = model_akida.evaluate(x_test, y_test.astype(np.int32))
print('Test accuracy after conversion:', accuracy)

# For non-regression purposes
assert accuracy > 0.96

# %% [3.3. - Show predictions for a single image]
# Test a single example
sample_image = 0
image = x_test[sample_image]
outputs = model_akida.predict(image.reshape(1, 28, 28, 1))
print('Input Label: %i' % y_test[sample_image])

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(x_test[sample_image].reshape((28, 28)), cmap=cm.Greys_r)
axarr[0].set_title('Class %d' % y_test[sample_image])
axarr[1].bar(range(10), outputs.squeeze())
axarr[1].set_xticks(range(10))
plt.show()

print(outputs.squeeze())


# %%
