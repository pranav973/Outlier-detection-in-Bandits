import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import math



def bern(mean):
  if np.random.random_sample() < mean:
    return 1
  else:
    return 0


def beta(M,est,delta_curr):
  if est == float("-inf"):
    return 0
  r = special.erfinv(1-(delta_curr/2.0))
  print(r)
  p = ((est*M)+((r**2)/2.0))/(M+(r**2))
  print(p)
  #print(est)
  return r*math.sqrt((p*(1-p))/float(M))

def beta_th(est_vec,k,conf,n):
  value = 0
  for t in range(n):
    var = ((k*est_vec[t]*conf[t])/(math.sqrt(np.std(est_vec))))+conf[t]
    value += var**2
  return (math.sqrt(value)/float(n))

def CLUCB(mean,delta,n,k):
  
  outlier_est = set()
  est_mean = np.zeros(n)
  confidence = np.zeros(n)
  arm_count = np.zeros(n)
  est_tilda = np.zeros(n)
  T = 0
  current_set = set(range(1,n+1))
  for t in current_set:
    sample = bern(mean[t-1])
    est_mean[t-1] = sample
  arm_count += 1

  T = n
  confidence = (2*np.log(((4*n)*(T**3))/delta))/(arm_count)
  
  while T < 10000:
    while True:
      copy_estmean = np.zeros(n)
      for l in range(n):
        copy_estmean[l] = est_mean[l]
      for y in outlier_est:
        copy_estmean[y-1] = float("-inf")
      best = np.where(copy_estmean == max(copy_estmean))[0][0]
     
      est_tilda = copy_estmean+confidence
      est_tilda[best] = copy_estmean[best]-confidence[best]
      best_tilda = np.where(est_tilda == max(est_tilda))[0][0]
     
      if est_tilda[best] == est_tilda[best_tilda]:
        
        break
      else:
        if confidence[best] > confidence[best_tilda]:
          
          sample = bern(mean[best])
          est_mean[best] = ((est_mean[best]*arm_count[best])+sample)/(arm_count[best]+1)
          arm_count[best] += 1
        else:
          
          sample = bern(mean[best_tilda])
          est_mean[best_tilda] = ((est_mean[best_tilda]*arm_count[best_tilda])+sample)/(arm_count[best_tilda]+1)
          arm_count[best_tilda] += 1
        T += 1
        confidence = (2*np.log(((4*n)*(T**3))/delta))/(arm_count)
    roal_conf = np.zeros(n)
    for a in range(n):
      if a+1 in outlier_est:
        roal_conf[a] = 0
      else:
        roal_conf[a] = beta(arm_count[a],est_mean[a],(0.6)/(((math.pi)**2)*(n+1)*(T**2)))

    threshold = np.mean(est_mean)+k*np.std(est_mean)
    threshold_confidence = beta_th(est_mean,k,roal_conf,n)
    print(threshold_confidence)
    if (threshold+threshold_confidence > est_mean[best]-roal_conf[best]) and (threshold+threshold_confidence < est_mean[best]+roal_conf[best]):
      outlier_est.add(best+1)
    elif (threshold-threshold_confidence < est_mean[best]+roal_conf[best]) and (threshold+threshold_confidence > est_mean[best]+roal_conf[best]):
      outlier_est.add(best+1)

      
    else:
      break
    print(outlier_est)
  #print(threshold)
  #print(threshold_confidence)
  #print(est_mean)
  #print(confidence)
  return (outlier_est,T)

'''clucb_T = []
clucb = []

k = [2,2.5,3]
for a in [20,30,40,50,75,100,120,150,175,200]:
  for x in k:
    posi2 = 0
    
    cluc_c = 0
    for t in range(10):
      means = 0.5*np.random.random(a)
      means[-1] = 0.999
      means[-2] = 0.999
      A = set()
      for r in range(a):
        if means[r] > np.mean(means)+(x*np.std(means)):
          A.add(r)
      
      #a1 = NRR(means,a,x)
      a2 = CLUCB(means,0.1,a,x)
     
      
      #if A == a1[0]:

        #posi1 += 1
      
      if A == a2[0]:
        print(A)
        print(a2[0])
        posi2 += 1
      
      #nrr_T.append(a1[1])
      cluc_c += a2[1]
      
    #nrr.append(posi1/10.0)
    clucb_T.append(cluc_c/10.0)
    
    clucb.append(posi2/10.0)
    
  posi2 = 0'''
 


    
      
    
    




