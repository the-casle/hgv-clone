import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import os
os.chdir(r'C:\Users\GreenFluids_VR\Documents\GitHub\CUDA__Project\code\python')
#os.chdir(r'C:\Users\justi\Documents\GitHub\CUDA__Project\code\python')
run_num = 7
number = 0

df=np.loadtxt(f"../data/Run {run_num}/trajNA187_PY_RK4_NN_{number}.txt")
#df=np.loadtxt(f"../data/trajNA187_PY_RK4_2.txt")
fig, axs = plt.subplots(5,figsize=(9,18),dpi=200)
#fig.suptitle(f"Episode {number}")
fig.tight_layout()
# df_=np.loadtxt(f"../run_test_h/tmp.38/traj_NA_model_fit.txt")
#df_10=np.loadtxt(f"./data/trajNA187_C.txt")
# altitude 
axs[0].plot(df[:,4]/1e3,df[:,3]/1e3,".-",label="187k")
# axs[0].plot(df_[:,4]/1e3,df_[:,3]/1e3,".",c="k",label="NA scale")
#axs[0].plot(df_10[:,4]/1e3,df_10[:,3]/1e3,c="k",label="417k")
axs[0].set_ylabel("Altitude [Km]")
axs[0].set_xlabel("DownRange [Km]")
axs[0].set_ylim([0, 100])
# heading angle 
axs[1].plot(df[:,0],-np.rad2deg(df[:, 1]),label="187k")
# axs[1].plot(df_[:,0],-np.rad2deg(df_[:, 1]),".",c="k",label="NA scale")
#axs[1].plot(df_10[:,0],-np.rad2deg(df_10[:, 1]),c="k",label="417k")
axs[1].set_ylabel(r"$\gamma$ [degs]")
axs[1].set_xlabel("time [s]")
axs[1].set_ylim([-7.5, 7.5])
# L/D
axs[2].plot(df[:,0],df[:,2]/1e3,label="187k")
# axs[2].plot(df_[:,0],df_[:,2]/1e3,".",c="k",label="NA scale")
#axs[2].plot(df_10[:,0],df_10[:,2]/1e3,c="k",label="417k")
axs[2].set_ylabel(r"vel [km/s]")
axs[2].set_xlabel("time [s]")
axs[2].set_ylim([0, 10])
# velocity
axs[3].plot(df[:,0],df[:,6],label="187k")
# axs[3].plot(df_[:,0],df_[:,6],".",c="k",label="NA scale")
#axs[3].plot(df_10[:,0],df_10[:,6],c="k",label="417k")
axs[3].set_ylabel(r"L/D [-]")
axs[3].set_xlabel("time [s]")
#axs[3].set_ylim([-5, 5])

#Plot angle of attack
axs[4].plot(df[:,0],np.rad2deg(df[:,5]),label="187k")
# axs[4].plot(df_[:,0]/60,1000*(9.81*df_[:,3]+0.5*df_[:,2]**2)*1e-9,".",c="k",label="NA")
#axs[4].plot(df_10[:,0],np.rad2deg(df_10[:,5]),c="k",label="417k")
axs[4].set_ylabel(r"AOA[deg]") 
axs[4].set_xlabel("time [s]")
# axs[4].plot(df[:,0],1000*(9.81*df[:,3]+0.5*df[:,2]**2)*1e-9,label="187k")
# # axs[4].plot(df_[:,0]/60,1000*(9.81*df_[:,3]+0.5*df_[:,2]**2)*1e-9,".",c="k",label="NA")
# axs[4].plot(df_10[:,0],1000*(9.81*df_10[:,3]+0.5*df_10[:,2]**2)*1e-9,c="k",label="417k")
# axs[4].set_ylabel(r"Total Energy [GJ]") 
# axs[4].set_xlabel("time [s]")
#axs[4].set_ylim([-60, 60])
for a in axs:
    a.ticklabel_format(useOffset=False)
    a.legend()