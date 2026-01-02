#required libraries
from src.load_data import load_subject
import matplotlib.pyplot as plt
from src.preprocess import normalize
from src.feature import extract_features
from src.windowing import sliding_window
from src.model import train_model
import numpy as np
from src.persistent_anomaly import compute_persistent_segments
import joblib

#-------Unsupervised physiological anomaly detection using wearable BVP signal----------

#load data
data=load_subject("data/S10.pkl")
BVP=data["signal"]["wrist"]["BVP"].flatten()
labels=data["label"]#-1,0,1,2....
#print(BVP.shape)# required 1D signal
#o/p->(351744,)


#normalization
norm_bvp=normalize(BVP)
#print(norm_bvp.shape)# required 1D shape
#o/p->(351744,)

#windowing
window,index=sliding_window(norm_bvp,640,128)#5 sec
#window=window[...,np.newaxis]#gives 3D shape , we required it when used simple features 
#print(window.shape)
#o/p->(2743, 640, 1)

#print(window[0].shape)
def comput_win_label(index,labels):
  win_labels=[]
  for i in index:
    i_1=min(i[0],len(labels)-1)
    j_1=min(i[1],len(labels))
    win_labels.append(np.bincount(labels[i_1:j_1]).argmax())
  win_labels=np.array(win_labels)
  return win_labels

window_label=comput_win_label(index,labels)

#print("unique len of windows:",np.unique(window_label))
#print("count label==1 :",np.sum(window_label==1))

#baseline_windows=(window[win_labels==1]).reshape(-1)

#------we are calculating rhythms so we need kurtosis and skew, for that window shape must be 2D---
#feature extraction
feature=extract_features(window)
#print("feature shape:",feature.shape)

#train model using isolation forest
x_train=feature[window_label==1]
#print("x_train shape :",x_train.shape)

assert x_train.ndim==2
def get_trained(x_train):
  model=train_model(x_train)
  joblib.dump(model,"bvp_iforest.pkl")
  return model
  
model=get_trained(x_train)

#score
Total_scores=model.decision_function(feature)#not binary o/p it contain graduation score (,,,,-0.05,-0.02,0,0.1,0.4,0.6,,,,)
training_score=model.decision_function(x_train)

threshold=np.percentile(training_score,2)
anomaly_flags=np.array(Total_scores<threshold)


'''persistent anomalies are the anomalies which stay consistent for some duration
win_size=640, sample rate=128,
take stride=320(50% overlap)

aaplyala atleast 10 sec sathi persistent check karaych ahe , 1w->5sec in our case'''
k=6# no of win*win_time/stride_time
persistent_flags = compute_persistent_segments(anomaly_flags, k)

segments = []
in_segment = False
for i, flag in enumerate(persistent_flags):
    if flag and not in_segment:
        start = i
        in_segment = True
    elif not flag and in_segment:
        segments.append((start, i))
        in_segment = False

if in_segment:
    segments.append((start, len(persistent_flags)))


y_pred=model.predict(feature)#binary o/p(-1,0,1)

'''print(len(Total_scores))
print(np.sum(anomaly_flags))
print(np.sum(persistent_flags))'''

def score_distribution(win_labels):
 #score distribution
 plt.hist(Total_scores[win_labels==1],bins=50,alpha=0.6,label="baseline")
 plt.hist(Total_scores[win_labels!=1],bins=50,alpha=0.6,label="stress")
 plt.xlabel("score")
 plt.ylabel("window_size")
 plt.title("Score Distribution")
 plt.legend()
 return plt.show()

def timeline_anomaly_score():
#timeline anomaly score
   plt.plot(Total_scores,label="Score")
   plt.axhline(threshold,color="r",label="Threshold",linestyle="--")# threshold chosen based on baseline score distribution (5th percentile)
   for s in segments:
      plt.axvspan(s[0],s[1],color="red",alpha=0.25,label="Persistent Anomalies")
   plt.legend()
   plt.xlabel("windows")
   plt.ylabel("score")
   plt.title("Timeline anomaly score")
   return plt.show()

#model flags temporal deviations in physiological signal as anomalies.



