import numpy as np
import matplotlib.pyplot as plt
import math

def bernoulli(mean):
  if np.random.uniform(0,1,1)[0] >= mean:
    return float(0)
  else:
    return float(1)
Time = np.zeros(10)
for s in range(5):
  for p in range(1,11):
    n = 100*p
    k = 2.5
    means = np.random.random(100*p)
    
    
   
    threshold = np.mean(means)+k*np.std(means)
    '''while np.min(np.abs(threshold-means)) <= 0.1:
      means = np.random.uniform(0,0.5,200)
      means[-1] = 1
      threshold = np.mean(means)+k*np.std(means)'''
    time = 1
    samples = 0
    delta = 0.1
    
    delta_par = (3*delta)/((n+4)*(math.pi**2))

    true_outliers = set()
    for x in range(n):
      if means[x] >= threshold:
        print(means[x])
        print(threshold)
        true_outliers.add(x+1)
    print(true_outliers)
      



    #arm_radius = np.zeros(n)
    seq_radius = 0
    #arm_radius = np.zeros(n)
    para_radius = 0
    #arm_count = np.zeros(n)
    threshold_count = 0
    estimated_mean = np.zeros(n)
    active_set = set(range(1,n+1))
    outliers = set()
    seq_count = 0
    current_estmean = 0
    current_eststd= 0
    current_threshold = 0
    sum_1 = 0
    sum_2 = 0
    product = 0


    rand_arm = np.random.randint(1,n+1)
    (sample1,sample2) = (bernoulli(means[rand_arm-1]),bernoulli(means[rand_arm-1]))
    threshold_count += 1
    time += 1
    current_estmean = sample1
    sum_1 = sample1
    sum_2 = sample2
    product = sample1*sample2
    current_threshold = current_estmean+k*current_eststd
    #arm_count += 1
    seq_count += 1
    time += 1
    estimated_mean = []
    for x in range(n):
      estimated_mean.append(bernoulli(means[x]))
    estimated_mean = np.array(estimated_mean)
    #print(estimated_mean)
    samples += 2*n+2

    seq_radius = ((np.log((time**2)/delta_par))/(2*seq_count))**0.5
    #arm_radius = ((np.log((time**2)/delta_par))/arm_count)**0.5
    current_U = min_U = current_eststd**2+(3*(((np.log((time**2)/delta_par))/(2*threshold_count))**0.5))

    para_radius = (1+1.414*k*3*(current_U**(-0.5)))*((np.log((time**2)/delta_par))/(2*threshold_count))**0.5


    while active_set:
      #print(para_radius)
      #print(seq_radius)
      #print(estimated_mean)
      time += 1
      if seq_radius <= para_radius:
        rand_arm = np.random.randint(1,n+1)
        (sample1,sample2) = (bernoulli(means[rand_arm-1]),bernoulli(means[rand_arm-1]))
        sum_1 += sample1
        sum_2 += sample2
        samples += 2
        product += sample1*sample2
        threshold_count += 1
        current_estmean = sum_1/threshold_count
        current_eststd = (abs((product/threshold_count)-((sum_1*sum_2)/(threshold_count**2))))**(0.5)
        current_threshold = current_estmean+k*current_eststd
        seq_radius = ((np.log((time**2)/delta_par))/(2*seq_count))**0.5
        #for p in active_set:
          #arm_radius[p] = ((np.log((time**2)/delta_par))/arm_count[p])**0.5
        current_U = current_eststd**2+(3*(((np.log((time**2)/delta_par))/(2*threshold_count))**0.5))
        min_U = min(current_U,min_U)
        para_radius = (1+(1.414*k*3*(min_U**(-0.5))))*(((np.log((time**2)/delta_par))/(2*threshold_count))**0.5)
      else:
        seq_count += 1
        seq_radius = ((np.log((time**2)/delta_par))/(2*seq_count))**0.5
        for x in active_set:
          sample = bernoulli(means[x-1])
          estimated_mean[x-1] = (estimated_mean[x-1]*(seq_count-1)+sample)/(seq_count)
        current_U = current_eststd**2+(3*(((np.log((time**2)/delta_par))/(2*threshold_count))**0.5))
        min_U = min(current_U,min_U)
        para_radius = (1+(1.414*k*3*(min_U**(-0.5))))*(((np.log((time**2)/delta_par))/(2*threshold_count))**0.5)
        samples += len(active_set)
      unactive = []
      
      for x in active_set:
        
        if estimated_mean[x-1]+seq_radius <= current_estmean+(k*current_eststd)-para_radius:
          unactive.append(x)
        elif estimated_mean[x-1]-seq_radius >= current_estmean+(k*current_eststd)+para_radius:
          unactive.append(x)
          outliers.add(x)
      for a in unactive:
        active_set.remove(a)
    Time[p-1] += samples 

print(Time/5)
  








    



    


















