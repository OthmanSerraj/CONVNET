from __future__ import print_function
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from random import randint
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from scipy import ndimage
from scipy import misc
import random
import time
import cv2
directory='Image données en sous dossiers'
taille=len(os.listdir(directory))
square_size=75
def get_square(img,size):
    if size==None:return img
    interpolation=cv2.INTER_AREA
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)
def vectgen(i,taille):
    tab=[0]*taille
    tab[i]=1
    return tab
os.system('cls')
print("="*37+"Leaf Recognition System"+"="*37)
def f(directory):
    targets = []
    features = []
    folders=os.listdir(directory)
    print("Nous allons charger "+str(taille)+" dossiers :")
    files =[]
    i=0
    for folder in folders:
        print(folder,end=", ")
        target=vectgen(i,taille)
        files=glob.glob(directory+'/'+folder+'/*.jpg')
        for file in files:
            tof=np.array(Image.open(file))
            features.append(get_square(tof,square_size))
            targets.append(target)
        i+=1
    return np.array(features),np.array(targets)
if 'Data' in locals():print("Nous avons déjà chargé notre Data!")
else:
    Data=f(directory)
    features,targets = Data
    print("*")
    print("\n")
    print("Data charger avec succès!!!")
varal=randint(0,len(features)-1)
imager=features[varal]
plt.imshow(imager, cmap="gray")
LL=targets[varal]
print(LL)
for u in range(taille):
    if targets[varal][u]==1:
        print(os.listdir(directory)[u])
        break
plt.show()
def g():
    X_train, X_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.05, random_state=42)
    return X_train, X_valid, y_train, y_valid
if 'Data_traité' in locals():print("Nous avons déjà traité notre Data!")
else:
    Data_traité=g()
    X_train, X_valid, y_train, y_valid = Data_traité
    print("Data traiter avec succès!!!")
def create_conv(prev, filter_size, nb):
    conv_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, int(prev.get_shape()[-1]), nb)))
    conv_b = tf.Variable(tf.zeros(nb))
    conv   = tf.nn.conv2d(prev, conv_W, strides=[1, 1, 1, 1], padding='SAME') + conv_b
    conv = tf.nn.relu(conv)
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return conv
x = tf.placeholder(tf.float32, (None, square_size, square_size, 3), name="x")
y = tf.placeholder(tf.float32, (None, taille), name="y")
def get_network(x,y):    
    dropout = tf.placeholder(tf.float32, (None), name="dropout")
    conv = create_conv(x, 8, 5)
    conv = create_conv(conv, 5, 11)
    print("Taille de sortie: ",conv.shape)
    #conv = create_conv(conv, 5, 23)
    # conv = create_conv(conv, 5, 47)
    # conv = create_conv(conv, 5, 97)
    flat = flatten(conv)
    fc1_W = tf.Variable(tf.truncated_normal(shape=(int(flat.get_shape()[1]), 512)))
    fc1_b = tf.Variable(tf.zeros(512))
    fc1   = tf.matmul(flat, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, taille)))
    fc3_b  = tf.Variable(tf.zeros(taille))
    logits = tf.matmul(fc1, fc3_W) + fc3_b
    softmax = tf.nn.softmax(logits)
    Network=True
    return y,logits,softmax,dropout
if 'Network' in locals():print("Nous avons déjà créé un Réseau!")
else:
    Network=True
    y,logits,softmax,dropout=get_network(x,y)
    print("Réseau créer avec succès!!!")
if 'Minimisation' in locals():print("Nous avons déjà créé les Variables de minimisation!")
else:
    Minimisation=True
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    predicted_cls = tf.argmax(softmax, axis=1)
    correct_prediction = tf.equal(predicted_cls, tf.argmax(y, axis=1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
    training_operation = optimizer.minimize(loss_operation)
    print("Toutes les variables ont été créer avec succès")
def firstuse():
    return os.path.isfile("mon modèle/checkpoint") 
if 'saver' in locals():print("Nous avons déjà chargé les Variables du model enregistré!")
elif firstuse():
    batch_size = 250
    saver = tf.train.Saver()
    sess =  tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess,"mon modèle/Model.ckpt")
elif not firstuse():
    batch_size = 300
    saver = tf.train.Saver()
    sess =  tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
def augmented_batch(batch):
    n_batch = []
    for img in batch:
        if random.uniform(0, 1) > 0.75:
            process_img = Image.fromarray(np.uint8(img.reshape(square_size, square_size, 3))).rotate(randint(-18000, 18000))
            n_img = np.array(process_img)
            n_batch.append(n_img.reshape(square_size, square_size, 3))
        else:
            n_batch.append(img)
    return n_batch
print("Lancement de l'entraînement dans: ")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")
i = 0
for epoch in range(0, 5000):
    print(">> Epoch: %s" % epoch)
    if epoch%5==0:
        saver = tf.train.Saver()
        save_path = saver.save(sess, "mon modèle/Model.ckpt")
        print("Modèle mis à jour avec succès!")
    indexs = np.arange(len(X_train))
    np.random.shuffle(indexs)
    X_train = X_train[indexs]
    y_train = y_train[indexs]
    for b in range(0, len(X_train), batch_size):
        batch = augmented_batch(X_train[b:b+batch_size])
        if i % 20 == 0:
            print("Accuracy [Train]:", sess.run(accuracy_operation, feed_dict={dropout: 1.0, x: batch, y: y_train[b:b+batch_size]}))            
        sess.run(training_operation, feed_dict={dropout: 0.8, x: batch, y: y_train[b:b+batch_size]})
        i += 1
    if epoch % 2 == 0:
        accs = []
        for b in range(0, len(X_valid), batch_size):
            accs.append(sess.run(accuracy_operation, feed_dict={dropout: 1., x: X_valid[b:b+batch_size], y: y_valid[b:b+batch_size]}))
        print("Accuracy [Validation]", np.mean(accs))