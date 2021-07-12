import numpy as np
import matplotlib.pyplot as plt
import math
from time import time


# Indexing function for Gaussian and Sub Gaussian bandits
def Index_fn(est_arr,c_alp,N_arr,t):
    return est_arr+((c_alp*np.log(t))/(2.0*N_arr))

# fix the standard deviation to 1.
t = 0
T = int(input())
N = int(input())

true_mean = [0.41,0.52,0.66,0.43,0.58,0.65,0.48,0.67,0.59,0.63]
true_mean.sort()
true_mean = np.array(true_mean)
print(true_mean)
est_mean = np.zeros(N)
N_arr = np.zeros(N)


for x in range(N):
    sample = np.random.normal(true_mean[x],1,1)
    est_mean[x] = sample
    N_arr[x] = 1
    t += 1


regret_arr = []
regret = 0
# sample all arms once in order to make sure N != 0
for x in range(N):
    regret += true_mean[-1]-true_mean[x]
    regret_arr.append(regret)
gap = 0

start1 = time()
# iterate until "T" time
while t <= T:
    upper = est_mean+((2*(N+2)*np.log(t))/N_arr)**(0.5) # lower bound in confidence interval
    lower = est_mean-((2*(N+2)*np.log(t))/N_arr)**(0.5) # upper bound in confidence interval
    gap_arr = [0 for x in range(N)]
    for p in range(N):
        gap_arr[p] = max(0,lower[p]-max([upper[l] for l in range(N) if l != p])) # Estimated minimum gap is always less than actual minimum gap
    
    gap = max(gap_arr)
    if gap == 0:
      c_alp = float("inf")
    else:

      c_alp = 256/gap
    Ind_arr = Index_fn(est_mean,min(c_alp,(np.log(t))**(0.5)),N_arr,t)
    arm_index = np.where(Ind_arr == max(Ind_arr))[0][0]
    #print(arm_index)
    regret += true_mean[-1]-true_mean[arm_index]
    
    #print(regret)
    regret_arr.append(regret)
    sample = np.random.normal(true_mean[arm_index],1,1)
    N_arr[arm_index] += 1 # update the arm count
    est_mean[arm_index] = (est_mean[arm_index]*(N_arr[arm_index]-1)+sample)/N_arr[arm_index] # update the estimate
    t += 1
end1 = time()

print("RBMLE time is "+str((end1-start1)/10000))
################################################################################################## UCB Algorithm #################################################################

# ep is number of episodes, k is the number of arms, n is time horizon, R is the cumulative regret until actual algorithm begins

def ucb(mean,k,alpha,n,ep,R):
	
  rearr = [0 for x in range(n-k)]
  for a in range(ep):
    regret = R
    est_mean = np.zeros(k)
    for t in range(k):
      est_mean[t] = np.random.normal(mean[t],1,1)
			
    arm_count = np.ones(k)
    for x in range(k+1,n+1):
      con = (alpha*np.log(x))/2.0
      con = np.sqrt(con)
      ucb = est_mean+con*np.power(arm_count,-0.5)
      arm_played = np.where(ucb == max(ucb))[0][0]
      arm_count[arm_played] += 1
      regret += mean[-1]-mean[arm_played]
      rearr[x-k-1] += regret
      sample = np.random.normal(mean[arm_played],1,1)
      est_mean[arm_played] = (float(arm_count[arm_played]-1)*est_mean[arm_played]+sample)/(arm_count[arm_played])
  
			
  return [t/ep for t in rearr]

########################################################################################## UCB Normal ################################################################################################

def ucb_normal(mean,k,n,ep,R):
	
  rearr = [0 for x in range(n-k)]
  
  for a in range(ep):
    squares = np.zeros(k)
    regret = R
    est_mean = np.zeros(k)
    for t in range(k):
      est_mean[t] = np.random.normal(mean[t],1,1)
      squares[t] = 2*est_mean[t]**2
			
    arm_count = 2*np.ones(k)
    for x in range(k+1,n+1):
      parity = False
      for y in range(k):
        if arm_count[y] <= int(8*np.log(x)):
          parity = True
          sample = np.random.normal(mean[y],1,1)
          squares[y] += sample**2
          arm_count[y] += 1
          est_mean[y] = (float(arm_count[y]-1)*est_mean[y]+sample)/(arm_count[y])
          regret += mean[-1]-mean[y]
          rearr[x-k-1] += regret
          break
      if parity:
        continue



          
      con = (np.log(x-1))
      con = np.sqrt(con)
      normal = est_mean+4*con*np.power((squares-(arm_count*(est_mean**2)) )/(arm_count*(arm_count-1)),0.5)
      
      arm_played = np.where(normal == max(normal))[0][0]
      arm_count[arm_played] += 1
      regret += mean[-1]-mean[arm_played]
      rearr[x-k-1] += regret
      sample = np.random.normal(mean[arm_played],1,1)
      squares[arm_played] += sample**2
      est_mean[arm_played] = (float(arm_count[arm_played]-1)*est_mean[arm_played]+sample)/(arm_count[arm_played])
      
  
			
  return [t/ep for t in rearr]

########################################################################### UCB-Tuned #########################################################################################

def MOSS(mean,k,n,ep,R):
	
  rearr = [0 for x in range(n-k)]
  for a in range(ep):
    regret = R
    est_mean = np.zeros(k)
    for t in range(k):
      est_mean[t] = np.random.normal(mean[t],1,1)
			
    arm_count = np.ones(k)
    moss = np.zeros(k)
    for x in range(k+1,n+1):
      
      for y in range(k):
        moss[y] = est_mean[y]+((max(np.log(x/(arm_count[y]*k))/arm_count[y],0))**(0.5))
      arm_played = np.where(moss == max(moss))[0][0]
      arm_count[arm_played] += 1
      regret += mean[-1]-mean[arm_played]
      rearr[x-k-1] += regret
      sample = np.random.normal(mean[arm_played],1,1)
      est_mean[arm_played] = (float(arm_count[arm_played]-1)*est_mean[arm_played]+sample)/(arm_count[arm_played])
  
			
  return [t/ep for t in rearr]

################################################################################################### Thompson Sampling ##########################################################

# For a Gaussian reward distribution take priori also a Gaussian with zero mean and high Variance. The posterior will also be Gaussian with updated parameters from wikipedia.
# after intital exploration, sample from posterior distribution and play the arm with highest sample. Update posterior parameters only for the arm played.


def thompson(mean,k,n,ep,R):
	
  rearr = [0 for x in range(n-k)]
 
  for a in range(ep):
    mean_par = np.zeros(k) # posterior means
    var_par = 1000*np.ones(k) # posterior variances
    regret = R
    est_reward = np.zeros(k)
    arm_count = np.zeros(k)
    for t in range(k):
      est_reward[t] = np.random.normal(mean[t],1,1)
      mean_par[t] = (1.0/(0.001+1.0))*(est_reward[t]+(mean_par[t]/var_par[t]))
      var_par[t] = (1.0/(0.001+1.0))
      arm_count[t] += 1
			
    
    for x in range(k+1,n+1):
      sampled = np.array([np.random.normal(mean_par[t],(var_par[t])**(0.5),1)[0] for t in range(k)])
      #print(sampled)
      arm_played = np.where(sampled == max(sampled))[0][0]
      #print(arm_played)
      arm_count[arm_played] += 1
      est_reward[arm_played] += np.random.normal(mean[arm_played],1,1)
      mean_par[arm_played] = (1.0/((1.0/var_par[arm_played])+arm_count[arm_played]))*(est_reward[arm_played]+(mean_par[arm_played]/var_par[arm_played]))
      var_par[arm_played] = (1.0/((1.0/var_par[arm_played])+arm_count[arm_played])) 
      #print(mean_par)
      #print(var_par)
      regret += mean[-1]-mean[arm_played]
      rearr[x-k-1] += regret
  
  
			
  return [t/ep for t in rearr]

avg_regret = [0]
thomp_regret = [0]
MOSS_regret = [0]
normal_regret = [0]
reg = 0
for x in range(10):
  reg += true_mean[-1]-true_mean[x]
  avg_regret.append(reg)
  thomp_regret.append(reg)
  MOSS_regret.append(reg)
  normal_regret.append(reg)
start2 = time()
new = ucb(true_mean,10,2.0,100000,10,reg)
end2 = time()
print("UCB time is "+str((end2-start2)/100000))
start3 = time()
new2 = thompson(true_mean,10,100000,10,reg)
end3 = time()
print("Thompson time is "+str((end3-start3)/100000))
start4 = time()
new3 = MOSS(true_mean,10,100000,10,reg)
end4 = time()
print("MOSS time is "+str((end4-start4)/100000))
start5 = time()
new4 = ucb_normal(true_mean,10,100000,10,reg)
end5 = time()
print("UCB_normal time is "+str((end5-start5)/100000))
avg_regret.extend(new)
thomp_regret.extend(new2)
MOSS_regret.extend(new3)
normal_regret.extend(new4)
for x in range(T+1):
  normal_regret[x] = (avg_regret[x]+thomp_regret[x])/2.0
#avg_regret.extend([ucb(true_mean,10,2.0,x,1,reg) for x in no_runs if x > 10])
   
x_axis = range(T+1)
plt.plot(x_axis,avg_regret,"b",label = "UCB")
plt.xlabel("T")
plt.ylabel("Regret")
plt.title("Regret vs Time")
plt.plot(x_axis,list(regret_arr),"r",label="RBMLE")
plt.plot(x_axis,thomp_regret,"g",label="TS")
plt.plot(x_axis,MOSS_regret,"y",label="MOSS")
plt.plot(x_axis,normal_regret,"orange",label="UCB_Normal")
plt.legend(loc="lower right")
plt.show()

    
    
