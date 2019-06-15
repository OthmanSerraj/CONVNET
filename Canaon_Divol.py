from __future__ import print_function
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt
import os
from random import randint
import tensorflow as tf
from tensorflow.contrib.layers import flatten
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
os.system('cls')
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
if 'saver' in locals():print("Nous avons déjà chargé les Variables du model enregistré!")
else:
    saver = tf.train.Saver()
    sess =  tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess,"mon modèle/Model.ckpt")
def isvoyh(leaf_name):
    voyh="AEOIUYH"
    pre=leaf_name[0]
    for c in voyh:
        if pre==c:return "d'"
    return "de "
image_path=sys.argv[1][2:]
new_imag=np.array(Image.open(image_path))
new_image=get_square(new_imag,square_size)
plt.imshow(new_image, cmap="gray")
pya=  sess.run(predicted_cls,feed_dict={x: [new_image]})
Lista=os.listdir(directory)
new_logits = sess.run(logits,{x: [new_image]})[0].astype(int)
moy=abs(np.mean(new_logits))
new_logits=np.array([logit/moy for logit in new_logits]).astype(int)
os.system('cls')
mine=0
for logit in new_logits:
    if mine>logit:mine=logit
new_logits=[int((logit+abs(mine))**2) for logit in new_logits]

sem=0
for logit in new_logits:sem+=logit

new_logits=[logit/sem for logit in new_logits]
# print(new_logits)
def softmaxxme(new_logits):
    # LLO=[]
    # for logit in new_logits:
    #     if logit<0:logit=0
    #     x=logit**2
    #     LLO+=[x]
    # suma=np.sum(LLO)
    # softz=[]
    # for llo in LLO:
    #     x=llo/suma
    #     # if x<0.001:softz+=[0]
    #     # else:
    #     softz+=[x]
    return new_logits 
def repourcent(new_logits):
    rep=softmaxxme(new_logits)
    for i in range(taille):
        pr=str(rep[i])
        Lx=Lista[i]
        if rep[i]>0:
            space=23-len(Lx)
            Lx+=" "*space
            print(Lx+": "+pr[:7])

l_name=Lista[int(pya)].split(" ")
leaf_name=""
for sq in l_name:
    leaf_name+=sq.capitalize()+" "
ts = os.get_terminal_size()
Phrase1="Leaf Recognition System"
lp1=len(Phrase1)
cl=ts.columns
x=(cl-lp1)//2
Phrase2="Ceci est une feuille "+isvoyh(leaf_name)+leaf_name
lp2=len(Phrase2)
y=(cl-lp2-6)//2
a,b=0,0
if 2*x<(cl-lp1):b=1
elif 2*y<(cl-lp2-6):a=1

print("="*((cl-lp1)//2)+Phrase1+"="*((cl-lp1)//2+b))
print("\n")
repourcent(new_logits)
print("\n")
print("***"+"="*((cl-lp2-6)//2+a)+Phrase2+"="*((cl-lp2-6)//2)+"***")
plt.show()