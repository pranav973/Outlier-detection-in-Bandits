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
    sample = np.random.normal(means[x],1,1)
    est_mean[x] = sample
    arm_count[x] += 1
  t += n
  alpha = (np.log(t))**(0.5)
  while True:
   
      
    
   
    copy_mean = np.copy(est_mean)
    for popa in range(n):
      if not is_good[popa]:
        copy_mean[popa] = float("-inf")
    #print(arm_count)
    RBMLE_score = copy_mean+((alpha*np.log(t))/(2*arm_count))
    check_core = copy_mean+(2*np.log((800*n)*(arm_count**2))/(arm_count))**(0.5)
    if np.amax(check_core,where=is_good,initial=0) < thresh:
      break
    curr = np.where(RBMLE_score == max(RBMLE_score))[0][0]
    sample = np.random.normal(means[curr],1,1)
    
    est_mean[curr] = ((est_mean[curr]*arm_count[curr])+sample)/(arm_count[curr]+1)
    arm_count[curr] += 1
    t  += 1
    if est_mean[curr]-(2*np.log((800*n)*(arm_count[curr]**2))/(arm_count[curr]))**(0.5) >= thresh:
      result.append(t)
      good.add(curr)
      
      is_good[curr] = False
    elif est_mean[curr]+(2*np.log((800*n)*(arm_count[curr]**2))/(arm_count[curr]))**(0.5) < thresh:
      
      is_good[curr] = False
    upper = est_mean+((2*np.log(t)*(n+2))/arm_count)**(0.5)
    lower = est_mean-((2*np.log(t)*(n+2))/arm_count)**(0.5)
    
    value = float("-inf")
    for h in range(n):
      if not is_good[h]:
        continue
      for z in range(n):
        if is_good[z] == False or (h == z):
          continue
        else:
          value = max(value,max(0,lower[h]-upper[z]))
    if value == 0:
      alpha = np.log(t)**(0.5)
    else:
      alpha = max(256/(value**2),np.log(t)**(0.5))



  result.append(t) 
  return result

for x in range(100):
  res = hdoc([0.1,0.1,0.1,0.35,0.45,0.55,0.65,0.9,0.9,0.9],10,0.5)
  print(x)
  t1.append(res[0])
  t2.append(res[1])
  t3.append(res[2])
  t4.append(res[3])
  t5.append(res[4])
  t6.append(res[5])
  

  
  
t1 = np.array(t1)
t2 = np.array(t2)
t3 = np.array(t3)
t4 = np.array(t4)
t5 = np.array(t5)
t6 = np.array(t6)

    
print((np.mean(t1),np.std(t1)))
print((np.mean(t2),np.std(t2)))
print((np.mean(t3),np.std(t3)))
print((np.mean(t4),np.std(t4)))
print((np.mean(t5),np.std(t5)))
print((np.mean(t6),np.std(t6)))


