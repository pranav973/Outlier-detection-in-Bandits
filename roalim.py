# -*- coding: utf-8 -*-
"""roalim.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1E4zy3f-5-4T9GpQS5S7Zt-XBjLXPfDzE
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Generating subgaussian means for the arms
n = int(input()) # number of arms
k = float(input())
delta = float(input())
count = 0
for r in range(10):

  mean = np.random.normal(0.3,0.075,n)
  # generate from an uniform distribution

  means = np.sort(mean)
  means[n-2] = 0.2*np.random.random()+0.8
  means[n-1] = 0.2*np.random.random()+0.8
  actual_median = means[n//2]
  actual_std = np.median(np.absolute(means-actual_median))
  outlier_set = set()
  for l in range(n):
    if means[l] > actual_median+k*actual_median:
      outlier_set.add(l+1)
  print(outlier_set)
  estimates = np.zeros(n)
  lower = np.zeros(n)
  upper = np.zeros(n)
  sampled = np.zeros(n)

  lower_AD = np.zeros(n)
  higher_AD = np.zeros(n)
  para_lower = 0
  para_upper = 0

  # ROALIM
  pulls = 0
  active_set = set([x for x in range(1,n+1)])
  median_set = set(active_set)
  AD_set = set(active_set)
  bound_set = set(active_set)


  while bound_set:
    pulls += 1
    for x in active_set:
      sample = np.random.normal(means[x-1],1,1)
      estimates[x-1] = ((estimates[x-1]*sampled[x-1])+sample)/(sampled[x-1]+1)
      sampled[x-1] += 1
    lower = estimates-np.sqrt(np.log(4.0*pulls*(sampled**2)/delta)/(2.0*sampled))
    upper = estimates+np.sqrt(np.log(4.0*pulls*(sampled**2)/delta)/(2.0*sampled))
    lower_median = np.median(lower)
    upper_median = np.median(upper)
    lower_AD = np.maximum(lower-upper_median,lower_median-upper)
    upper_AD = np.maximum(upper-lower_median,upper_median-lower)
    L_AD = np.median(lower_AD)
    U_AD = np.median(upper_AD)
    para_lower = lower_median+k*L_AD
    para_upper = upper_median+k*U_AD
    set_1 = set()
    set_2 = set()
    set_3 = set()
    for t in range(n):
      if lower[t] < lower_median:
        if upper[t] < lower_median:
          continue
        else:
          set_1.add(t+1)
      else:
        if upper_median < lower[t]:
          continue
        else:
          set_1.add(t+1)
      if lower_AD[t] < U_AD:
        if upper_AD[t] < L_AD:
          continue
        else:
          set_2.add(t+1)
      else:
        if U_AD < lower_AD[t]:
          continue
        else:
          set_2.add(t+1)
      if lower[t] < para_lower:
        if upper[t] < para_upper:
          continue
        else:
          set_1.add(t+1)
      else:
        if para_upper < lower[t]:
          continue
        else:
          set_3.add(t+1)
    median_set = set_1.intersection(median_set)
    AD_set = set_2.intersection(AD_set)
    bound_set = set_3.intersection(bound_set)
    active_set = median_set.union(AD_set,bound_set)

  outlier = set()
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
print(count)
'''print(outlier)
print(pulls)
print(means)
print(estimates)
print(sampled)'''