from body_temp import norm_slope, temp_stable
from BVP import persistent_flags, anomaly_flags, Total_scores
import numpy as np
from fusion import bvp_temp_fuse
import json

fusion_outputs=bvp_temp_fuse(norm_slope,temp_stable,Total_scores,anomaly_flags)
fusion_outputs=np.array(fusion_outputs)
print(fusion_outputs[0])

def to_python(obj):
    if isinstance(obj,(np.integer,)):
        return int(obj)
    elif isinstance(obj,(np.floating,)):
        return float(obj)
    elif isinstance(obj,(np.bool,)):
        return bool(obj)
     
    return obj  

clean_op=[]
for row in fusion_outputs:
    clean_op.append({k:to_python(v) for k,v in row.items()})

with open("result.json","w") as f:
    json.dump(clean_op,f,indent=2)

