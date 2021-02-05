from keras.optimizers import SGD
import tensorflow as tf
from keras.utils import plot_model

#Core
import numpy as np
import pandas as pd
import math
import time
import random

#Visual
import matplotlib.pyplot as plt

#utils
from sklearn import preprocessing
from sklearn.metrics import make_scorer, accuracy_score,f1_score,roc_auc_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split



#consts
np.random.seed(800)
start=0
class_name="class"
test_rate =0.2
data_rate =1
score = accuracy_score
n_classes=0
scorer = make_scorer(accuracy_score)
dropout=True
look_back=1

#### FEATURES
n_layers=10
layer_size=50
n_epochs=10

#functions
def fix_time():
    global start
    start = time.time()

def elapsed():
    global start
    end = time.time()
    return end - start

def countClasses(matrix):
    global n_classes
    length=len(matrix)
    values=[]
    for s in range(0,length):
        values.append(matrix.iloc[s])
    n_classes=len(set(values))


def readCSV(name):
    global n_classes
    print("-----------------------------------")
    print("DB: ",name)
    pd.set_option("display.max_columns",500)
    #get data
    df = pd.read_csv(name+".csv")
    print(df.shape)
    print(df.describe())
    #sample data
    df = df.sample(frac=data_rate, replace=False,random_state=0)
    df = df.drop_duplicates()
    df=df.dropna(axis=0)
    features = list(df.head(0)) 
    colection = []
    names =[]
    for f in features:
        if df[f].dtype =='O' and f!=class_name :
            colection.append(pd.get_dummies(df[f],prefix=f).iloc[:,1:])
            names.append(f)
    if(len(colection)>0):
        df=df.drop(names,axis=1)
        print(df.shape)
        concatdf  =pd.concat(colection,axis =1)
        df = pd.concat([df,concatdf],axis=1)
        df.shape
    print(df.shape)
    #report_file.write("data size: "+str(df.shape)+"\n")
    print("Dataset size: ",str(df.shape))
    #get class distribuition
    target_counts = df[class_name].value_counts()
    print(max(target_counts)/sum(target_counts)  )
    #report_file.write("portion of class: "+str(max(target_counts)/sum(target_counts))+"\n")
    print("Class Distribution: ", str(max(target_counts)/sum(target_counts)))
    #reduce to featureset and class
    x_all = df.drop([class_name],axis=1)
    y_all = df[class_name]
    countClasses(y_all)
    #normalize
    x_all = preprocessing.normalize(x_all)
    #print head
    print(df.head(0))
    #generate train and test_set
    x_train, x_test, y_train, y_test = train_test_split(x_all,y_all,test_size =test_rate,stratify=y_all,random_state=0)
    return x_train, x_test, y_train, y_test

def model(x_train, x_test, y_train, y_test):
    global n_classes
    input_size = x_train.shape[1]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(input_size,)))
    for i in range(0,n_layers):
        model.add(tf.keras.layers.LSTM(layer_size, input_shape=(input_size, look_back)))
    if dropout:
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(optimizer='adam',
              #loss=['sparse_categorical_crossentropy'],
              #metrics=['accuracy'])
    history=model.fit(x_train, y_train, epochs=n_epochs)
    
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    score=model.evaluate(x_test, y_test)
    print("SCORE: ",score)




if __name__ == "__main__":
    print("Insert file name w/o extension: ")
    db=input()
    x_train, x_test, y_train, y_test = readCSV(db)
    print("N CLASSES: ", n_classes)
    report_file=open(db+"_report.txt","w")
    fix_time()
    model(x_train, x_test, y_train, y_test)
    print("Elapsed time: ",elapsed())