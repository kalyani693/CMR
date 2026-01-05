from src.preprocess import normalize
from src.feature import extract_features
from src.windowing import sliding_window
import numpy as np
from src.persistent_anomaly import compute_persistent
from src.feature_temp import feature_extraction
from fusion import bvp_temp_fuse,persist_data
import joblib
from main import clean_fusion_op
from BVP import timeline_anomaly_score,persistent_segments
import json
 
 
  

'''#for temp
fs=8# sample rate
duration=300 #seconds
n=fs*duration

#normal segment
temp=36.5+0.02*np.random.randn(n)

#trend(slow rise)
trend_start=100*fs
temp[trend_start:]+=np.linspace(0,0.08,n-trend_start)

#step change
step_start=200*fs
temp[step_start:]+=0.6

#for BVP
fs_bvp=128 #sample rate
n_bvp=fs_bvp*duration

t=np.arange(n_bvp)/fs_bvp
bvp=np.sin(2*np.pi*1.2*t)+0.05*np.random.randn(n_bvp)
#inject anomaly ( amplitude change)
factor=fs_bvp//fs
bvp[step_start*factor:(step_start+20)*factor]*=2'''

def test_model(bvp,temp): 
 #normalization
  norm_bvp=normalize(bvp)

#print(norm_bvp.shape, norm_temp.shape)#(38400,) (2400,)

#windowing
  win_bvp,idx_bvp=sliding_window(norm_bvp,640,128)
  win_temp,idx_temp=sliding_window(temp,40,8)
  win_temp=win_temp[...,np.newaxis]

#features
  fe_temp=feature_extraction(win_temp)
  fe_bvp=extract_features(win_bvp)
  fe_bvp=fe_bvp.reshape(fe_bvp.shape[0],-1)
#print(fe_bvp.shape)
  model =joblib.load("bvp_iforest.pkl")

  slope_temp=[]
  for i in fe_temp:
    slope_temp.append(i[1])

  n_slope=normalize(np.array(slope_temp))
  temp_stable=np.where((n_slope>=-1)&(n_slope<=1),1,0)# sta=1, unsta=0

  total_score=model.decision_function(fe_bvp)
  threshold=np.percentile(total_score,2)
  anomaly_flag=np.where(total_score<threshold,1,0)
#print("lenghts:" ,len(n_slope), len(temp_stable),len(total_score),len(anomaly_flag))

  fusion_result=bvp_temp_fuse(n_slope,temp_stable,total_score,anomaly_flag)
  clean_fussion_result=clean_fusion_op(fusion_result)
#print(clean_fussion_result[0])

  persistent_flags=compute_persistent(anomaly_flag,2)#k=2
  segments=persistent_segments(persistent_flags)
 
  timeline_anomaly_score(total_score,threshold,segments)
  persistent_anomaly_data=persist_data(segments)
  return clean_fussion_result,persistent_anomaly_data
 
def save_json(clean_fussion_result,persistent_data):
  data_to_save={
       "persistent_anomaly":persistent_data,
       "Window_wise_data":clean_fussion_result
      }  
  with open("result1.json","w") as f:
    json.dump(data_to_save,f,indent=2)  

