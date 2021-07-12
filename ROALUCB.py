import numpy as np
import matplotlib.pyplot as plt
import math

# Generating subgaussian means for the arms
n = int(input()) # number of arms
k = float(input())
delta = float(input())
count = 0


def fun(x):
  return x[0]
def argmax(arr):
  index = -1
  value = float("-inf")
  for x in range(len(arr)):
    if arr[x][0] > value:
      value = arr[x][0]
      index = x
  return arr[index][1]
def argmin(arr):
  index = -1
  value = float("inf")
  for x in range(len(arr)):
    if arr[x][0] < value:
      value = arr[x][0]
      index = x
  return arr[index][1]

for r in range(1):

  mean = np.zeros(n)
  # generate from an uniform distribution
  for x in range(15):
    mean[x] = x*0.142857
  mean[15] = 2.837+0.3
  mean[16] = 2.837+0.5

  means = np.sort(mean)
  #means[n-2] = np.random.random()+1.5
  #means[n-1] = np.random.random()+1.5
  print(means)
  def theoretical_sample_complexity(para,delt):
    mean_i = np.absolute(2.837-means)
    min_para = np.min(mean_i)
    median = np.median(means)
    median_i = np.absolute(median-means)
    
    AD_median = np.median(median_i)
    AD_i = np.absolute(median_i-AD_median)
    

    result = np.zeros(n)
    for x in range(n):
      result[x] = max(min_para,min(AD_i[x],mean_i[x],median_i[x]))
    
    count = para*(k**2)*np.sum(np.log((n*k)/(delt*result))/(result**2))
    return count
  actual_median = means[n//2]
  actual_std = np.median(np.absolute(means-actual_median))
  outlier_set = set()
  for l in range(n):
    if means[l] > actual_median+k*actual_std:
      outlier_set.add(l+1)
  print(outlier_set)
  estimates = np.zeros(n)
  lower = np.zeros(n)
  upper = np.zeros(n)
  sampled = np.zeros(n)
  
  lower_AD = np.zeros(n)
  higher_AD = np.zeros(n)
  
  # ROALIM
  pulls = 1
  active_set = set([x for x in range(1,n+1)])
  median_set = set(active_set)
  AD_set = set(active_set)
  bound_set = set(active_set)


  while bound_set:
    
    pulls += len(active_set)
    for x in active_set:
      sample = np.random.normal(means[x-1],0.5,1)
      estimates[x-1] = ((estimates[x-1]*sampled[x-1])+sample)/(sampled[x-1]+1)
      sampled[x-1] += 1
    lower = estimates-np.sqrt(np.log(4.0*n*(sampled**2)/delta)/(2.0*sampled))
    upper = estimates+np.sqrt(np.log(4.0*n*(sampled**2)/delta)/(2.0*sampled))
    lower_median = np.median(lower)
    upper_median = np.median(upper)

    lower_AD = np.maximum(lower-upper_median,lower_median-upper)
    upper_AD = np.maximum(upper-lower_median,upper_median-lower)

    
    esti =  [(estimates[x],x+1) for x in range(n)]
    esti.sort(key=fun,reverse=True)
    AD_esti = [((lower_AD[x]+upper_AD[x])/2,x+1) for x in range(n)]
    AD_esti.sort(key=fun,reverse=True)
    Lower = [(lower[esti[p][1]-1],esti[p][1]) for p in range(n)]
    Upper = [(upper[esti[p][1]-1],esti[p][1]) for p in range(n)]
    Lower_AD = [(lower_AD[AD_esti[p][1]-1],AD_esti[p][1]) for p in range(n)]
    Upper_AD = [(upper_AD[AD_esti[p][1]-1],AD_esti[p][1]) for p in range(n)]



    median_set = set()
    median_set.add(argmin(Lower[:(n+1)//2]))
    median_set.add(argmin(Lower[:(n+3)//2]))
    median_set.add(argmax(Upper[(n+3)//2:]))
    median_set.add(argmax(Upper[(n+5)//2:]))

    AD_set = set()
    AD_set.add(argmin(Lower_AD[:(n+1)//2]))
    AD_set.add(argmin(Lower_AD[:(n+3)//2]))
    AD_set.add(argmax(Upper_AD[(n+3)//2:]))
    AD_set.add(argmax(Upper_AD[(n+5)//2:]))

    bound_set = set()
    lower_para = lower_median+k*np.median(lower_AD)
    upper_para = upper_median+k*np.median(upper_AD)

    current_outliers = set()
    for x in range(n):
      if estimates[x] > (lower_para+upper_para)/2:
        current_outliers.add(x+1)
    
    new_set = set()
    for x in range(n):
      if lower[x] < lower_para:
        if upper[x] > lower_para:
          new_set.add(x+1)
      else:
        if upper_para > lower[x]:
          new_set.add(x+1)
    one = argmax([(upper[x-1],x) for x in range(1,n+1)])
    two = argmax([(lower[x-1],x) if x in current_outliers else (float("-inf"),x) for x in range(1,n+1)])
    if one in new_set:
      bound_set.add(one)
    if two in new_set:
      bound_set.add(two)


    active_set = median_set.union(AD_set,bound_set)
  print(current_outliers)
  print(estimates)
  print(pulls)
  print(theoretical_sample_complexity(10,0.1))


  '''outlier = set()
  est_median = np.median(estimates)
  est_std = np.median(np.absolute(estimates-est_median)) 
  for t in range(n):
    if estimates[t] > est_median+k*est_std:
      outlier.add(t+1)
  if outlier == outlier_set:
    count += 1
  print(outlier)
  print(outlier_set)
  print(estimates)
  print(means)
print(count)'''
'''print(outlier)
print(pulls)
print(means)
print(estimates)
print(sampled)'''



    
      


  
    

  
