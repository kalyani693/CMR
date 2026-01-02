#required libraries
from src.load_data import load_subject
import matplotlib.pyplot as plt
from src.preprocess import normalize
import numpy as np
from src.windowing import sliding_window
from src.feature_temp import feature_extraction

'''anomaly in temp signal
not instant sprikes, sudden deviation, abnormal slope

temp-> slow response physiological signal

BVP->main
Temp->side role play
combination->better result'''

#load data
data=load_subject("data/S10.pkl")
temp=data["signal"]["wrist"]["TEMP"].flatten()
#print(temp.shape)#1D->(21984,)

'''plt.plot(temp)
plt.title("Temperature( Raw )")
plt.show()'''

#windowing
window,idx=sliding_window(temp,40,8)#5sec
#print(window.shape)

#feature
feature=feature_extraction(window)
#print(feature)
slope=[]
for i in feature:
    slope.append(i[1])

norm_slope=normalize(np.array(slope))

temp_stable=np.where((norm_slope>=-1)&(norm_slope<=1),1,0)# stable=1 , unstable=0
#print(np.sum(temp_stable)) #o/p->1971




