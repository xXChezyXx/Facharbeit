import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255
x_train = x_train.reshape(len(x_train),28,28,1)
x_test = x_test.reshape(len(x_test),28,28,1)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(3, 3, input_shape=(28,28,1), padding="valid", activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, 3, padding="valid", activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01), metrics=['accuracy'])
fit = model.fit(x_train,y_train, epochs=200, shuffle=True, validation_split= 0.3)
plt.plot(fit.history['accuracy'])
plt.plot(fit.history['val_accuracy'])
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoche")
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

test_loss, test_acc = model.evaluate(x_test,y_test)
print(test_loss)
print(test_acc)
