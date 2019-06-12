#!/usr/bin/env python

import tensorflow as tf

# get datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

i1_train, i2_train = y_train < 5, y_train >= 5
x1_train, x2_train = x_train[i1_train, ...], x_train[i2_train, ...]
y1_train, y2_train = y_train[i1_train, ...], y_train[i2_train, ...] - 5

i1_test, i2_test = y_test < 5, y_test >= 5
x1_test, x2_test = x_test[i1_test], x_test[i2_test]
y1_test, y2_test = y_test[i1_test], y_test[i2_test] - 5

# build model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

# open results file
outfile = open('results.csv', 'w')
print('dataset,optimizer,stage,accuracy', file=outfile)

# train and test with sgd
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x1_train, y1_train, epochs=5)
print(model.evaluate(x1_test, y1_test))
print('1,sgd,pre,{0:.4f}'.format(model.evaluate(x1_test, y1_test)[1]), file=outfile)
print('2,sgd,pre,{0:.4f}'.format(model.evaluate(x2_test, y2_test)[1]), file=outfile)
model.fit(x2_train, y2_train, epochs=5)
print('1,sgd,post,{0:.4f}'.format(model.evaluate(x1_test, y1_test)[1]), file=outfile)
print('2,sgd,post,{0:.4f}'.format(model.evaluate(x2_test, y2_test)[1]), file=outfile)

# train and test with adam
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x1_train, y1_train, epochs=5)
print('1,adam,pre,{0:.4f}'.format(model.evaluate(x1_test, y1_test)[1]), file=outfile)
print('2,adam,pre,{0:.4f}'.format(model.evaluate(x2_test, y2_test)[1]), file=outfile)
model.fit(x2_train, y2_train, epochs=5)
print('1,adam,post,{0:.4f}'.format(model.evaluate(x1_test, y1_test)[1]), file=outfile)
print('2,adam,post,{0:.4f}'.format(model.evaluate(x2_test, y2_test)[1]), file=outfile)

# close results file
outfile.close()
