import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import math


# weight const function

def w(n,k,delta):
  val = 0
  val += (1+(k*math.sqrt(n-1)))/(math.sqrt(n))
  val += k/math.sqrt((2.0)*np.log((math.pi**2)*(n**3)/(6*delta)))
  return ((n-1)/val)**(2.0/3)

# sampling from bernoulli bandits

def bern(mean):
  if np.random.random_sample() < mean:
    return 1
  else:
    return 0


# confidence length function
def beta(M,est,delta_curr):
  r = special.erfinv(1-(delta_curr/2.0))
  p = ((est*M)+((r**2)/2.0))/(M+(r**2))
  return r*math.sqrt((p*(1-p))/float(M))

# function for confidence length of threshold
def beta_th(est_vec,k,conf,n):
  value = 0
  for t in range(n):
    var = ((k*est_vec[t]*conf[t])/(math.sqrt(np.std(est_vec))))+conf[t]
    value += var**2
  return (math.sqrt(value)/float(n))







# we take delta = 0.1 for the complete exercise. For naive round robbins we will terminate loop when doesn't change for 10 iterations
# implementing naive round robbins  

def NRR(means,n,k):
  estimate = np.zeros(n)
  T = 0
  confidence = np.zeros(n)
  thet_est = 0
  A = set()
  m = np.zeros(n)
  for i in range(n):
    estimate[i] += bern(means[i]) 
  m += 1
  T += n
  delta_curr = (0.6)/(((math.pi)**2)*(n+1)*(T**2))
  thet_est = np.mean(estimate)+(k*np.std(estimate))
  for t in range(n):
    confidence[t] = beta(m[t],estimate[t],delta_curr)
  for t in range(n):
    if estimate[i] > thet_est:
      if (estimate[i]-confidence[i]) < (thet_est+beta_th(estimate,k,confidence,n)):
        A.add(i)
    else:
      if (estimate[i]+confidence[i]) > (thet_est-beta_th(estimate,k,confidence,n)):
        A.add(i)

  count = 0
  i = 0
  A_old = set(A)
  while count != 5 and T < 10000000 and A:
    i = (i%n)
    A = set()
    estimate[i] = (m[i]*estimate[i]+bern(means[i]))/(m[i]+1)
    m[i] += 1
    T += 1
    delta_curr = (0.6)/(((math.pi)**2)*(n+1)*(T**2))
    for t in range(n):
      confidence[t] = beta(m[t],estimate[t],delta_curr)
    thet_est = np.mean(estimate)+(k*np.std(estimate))
    for t in range(n):
      if estimate[i] > thet_est:
        if (estimate[i]-confidence[i]) < (thet_est+beta_th(estimate,k,confidence,n)):
          A.add(i)
      else:
        if (estimate[i]+confidence[i]) > (thet_est-beta_th(estimate,k,confidence,n)):
          A.add(i)
    if A == A_old:
      count += 1
    else:
      count = 0
    A_old = set(A)
    i += 1
  result = set()
  for a in range(n):
    if estimate[a] > (np.mean(estimate)+(k*np.std(estimate))):
      result.add(a)
  return result,T


# fucntion for iterative round robbins

def IRR(means,n,k):
  
  estimate = np.zeros(n)
  T = 0
  confidence = np.zeros(n)
  thet_est = 0
  A = set()
  m = np.zeros(n)
  for i in range(n):
    estimate[i] += bern(means[i]) 
  m += 1
  T += n
  delta_curr = (0.6)/(((math.pi)**2)*(n+1)*(T**2))
  thet_est = np.mean(estimate)+(k*np.std(estimate))
  for t in range(n):
    confidence[t] = beta(m[t],estimate[t],delta_curr)
  for t in range(n):
    if estimate[i] > thet_est:
      if (estimate[i]-confidence[i]) < (thet_est+beta_th(estimate,k,confidence,n)):
        A.add(i)
    else:
      if (estimate[i]+confidence[i]) > (thet_est-beta_th(estimate,k,confidence,n)):
        A.add(i)

  count = 0
  i = 0
  
  while A and T < 10000000:
    
      
    i = (i%n)
    A = set()
    estimate[i] = (m[i]*estimate[i]+bern(means[i]))/(m[i]+1)
    m[i] += 1
    T += 1
    delta_curr = (0.6)/(((math.pi)**2)*(n+1)*(T**2))
    for t in range(n):
      confidence[t] = beta(m[t],estimate[t],delta_curr)
    thet_est = np.mean(estimate)+(k*np.std(estimate))
    for t in range(n):
      if estimate[t] > thet_est:
        if (estimate[t]-confidence[t]) < (thet_est+beta_th(estimate,k,confidence,n)):
          A.add(t)
      else:
        if (estimate[t]+confidence[t]) > (thet_est-beta_th(estimate,k,confidence,n)):
          A.add(t)
    
    i += 1
  result = set()
  for a in range(n):
    if estimate[a] > (np.mean(estimate)+(k*np.std(estimate))):
      result.add(a)
  return result,T


# function for weighted round robin

def WRR(means,n,k,c):
  
  estimate = np.zeros(n)
  T = 0
  confidence = np.zeros(n)
  thet_est = 0
  A = set()
  m = np.zeros(n)
  for i in range(n):
    estimate[i] += bern(means[i]) 
  m += 1
  T += n
  delta_curr = (0.6)/(((math.pi)**2)*(n+1)*(T**2))
  thet_est = np.mean(estimate)+(k*np.std(estimate))
  for t in range(n):
    confidence[t] = beta(m[t],estimate[t],delta_curr)
  for t in range(n):
    if estimate[i] > thet_est:
      if (estimate[i]-confidence[i]) < (thet_est+beta_th(estimate,k,confidence,n)):
        A.add(i)
    else:
      if (estimate[i]+confidence[i]) > (thet_est-beta_th(estimate,k,confidence,n)):
        A.add(i)

  count = 0
  i = 0
  weigh = np.zeros(n)
  while A and T < 10000000:
    
      
    i = (i%n)
    weigh[i] += c
    A = set()
    estimate[i] = (m[i]*estimate[i]+bern(means[i]))/(m[i]+1)
    m[i] += 1
    T += 1
    delta_curr = (0.6)/(((math.pi)**2)*(n+1)*(T**2))
    for t in range(n):
      confidence[t] = beta(m[t],estimate[t],delta_curr)
    thet_est = np.mean(estimate)+(k*np.std(estimate))
    for t in range(n):
      if estimate[i] > thet_est:
        if (estimate[i]-confidence[i]) < (thet_est+beta_th(estimate,k,confidence,n)):
          A.add(i)
      else:
        if (estimate[i]+confidence[i]) > (thet_est-beta_th(estimate,k,confidence,n)):
          A.add(i)
    while i in A and weigh[i] > m[i]:
      A = set()
      estimate[i] = (m[i]*estimate[i]+bern(means[i]))/(m[i]+1)
      m[i] += 1
      T += 1
      delta_curr = (0.6)/(((math.pi)**2)*(n+1)*(T**2))
      for t in range(n):
        confidence[t] = beta(m[t],estimate[t],delta_curr)
      thet_est = np.mean(estimate)+(k*np.std(estimate))
      for t in range(n):
        if estimate[i] > thet_est:
          if (estimate[i]-confidence[i]) < (thet_est+beta_th(estimate,k,confidence,n)):
            A.add(i)
        else:
          if (estimate[i]+confidence[i]) > (thet_est-beta_th(estimate,k,confidence,n)):
            A.add(i)
    
    i += 1
  result = set()
  for a in range(n):
    if estimate[a] > (np.mean(estimate)+(k*np.std(estimate))):
      result.add(a)
  return result,T



# Iterative best arm identifying algorithm

#def IBA(means,n,k):

'''wi = []
ir = []
for p in range(1,11):
  mean = np.random.random(100*p)
  c1 = 0
  c2 = 0
  for a in range(2):
    c1 += IRR(mean,100*p,2.5)[1]
    c2 += WRR(mean,100*p,2.5,w(10*p,2.5,0.1))[1]
  wi.append(c2/2)
  ir.append(c1/2)
  print(wi)
  print(ir)

print(wi)
print(ir)'''
  

  


      







    


    

      
      
      
    
    

        



    




  



