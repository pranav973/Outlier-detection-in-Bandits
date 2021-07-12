import numpy as np
import matplotlib.pyplot as plt
import math



def bern(mean):
  if np.random.random(1) > mean:
    return 0
  else:
    return 1


t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
t6 = []


def hdoc(means,n,thresh):
  good = set()
  t = 0
  est_mean = np.zeros(n)
  arm_count = np.zeros(n)
  is_good = [True for x in range(n)]
  result = []
  for x in range(n):
    sample = bern(means[x])
    est_mean[x] = sample
    arm_count[x] += 1
  t += n
  while True:
   
      
    #print(arm_count)
    Ucb_score = est_mean+((np.log(t))/(2*arm_count))**(0.5)
    check_core = est_mean+(np.log((400*n)*(arm_count**2))/(2*arm_count))**(0.5)
    if np.amax(check_core,where=is_good,initial=0) < thresh:
      break
    curr = np.where(Ucb_score == max(Ucb_score))[0][0]
    sample = bern(means[curr])
    est_mean[curr] = ((est_mean[curr]*arm_count[curr])+sample)/(arm_count[curr]+1)
    arm_count[curr] += 1
    t  += 1
    if est_mean[curr]-(np.log((400*n)*(arm_count[curr]**2))/(2*arm_count[curr]))**(0.5) >= thresh:
      result.append(t)
      good.add(curr)
      est_mean[curr] = float("-inf")
      is_good[curr] = False
    elif est_mean[curr]+(np.log((400*n)*(arm_count[curr]**2))/(2*arm_count[curr]))**(0.5) < thresh:
      est_mean[curr] = float("-inf")
      is_good[curr] = False
  result.append(t) 
  return result

'''for x in range(100):
  res = hdoc([0.36,0.34,0.469,0.465,0.537],5,0.5)
  
  t1.append(res[0])
  t2.append(res[1])'''
  
  

    

