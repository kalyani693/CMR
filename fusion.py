from body_temp import norm_slope, temp_stable
from BVP import persistent_flags, anomaly_flags, Total_scores
import numpy as np

def bvp_temp_fuse(norm_slope,temp_stable,Total_scores,anomaly_flags):
    result=[]
    for i in range(len(anomaly_flags)):
        if anomaly_flags[i]==1 and temp_stable[i]==0:
            decision="Confirmed_anomaly"
        elif anomaly_flags[i]==1 and temp_stable[i]==1:
            decision="Possible_anomaly"
        else:
            decision="Normal"

        result.append({"window_id":i,
                       "bvp_score":Total_scores[i],
                       "bvp_anomaly":bool(int(anomaly_flags[i])),
                       "temp_slope":norm_slope[i],
                       "temp_stable":bool(int(temp_stable[i])),
                       "final_decision":decision}) 

    return result   


  
  
