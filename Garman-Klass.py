import numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy.stats import t,lognorm
from sklearn.cluster import KMeans

data = load_data(fname,index_col='Date')    #Write your own!!!
def GarmanKlassEstimator(data):
    v = []
    for i in data.index.values:
        temp = data.loc[i]
        hi,lo,o,c = temp.High,temp.Low,temp.Open,temp.Close
        v.append(np.sqrt(.5*np.log(hi/lo)**2-((2*np.log(2)-1)*np.log(c/o)**2)))
    data['Volatility'] = v

GarmanKlassEstimator(data)

v = data.Volatility.values     #our estimated volatility
km = KMeans(3)     #KMeans with n_clusters=3 for 3 volatility levels
km.fit(v.reshape(-1,1))     #reshape for 1-dimensional data
past_levels = km.labels_     #The levels predicted by the KMeans algo

#Now to make the transition matrix
tm = np.zeros((3,3))     #3x3 matrix of zeros for our 3 states
for i in range(1,len(past_levels)):
    a,b = past_levels[i-1],past_levels[i]
    tm[a][b]+=1
t_matrix = (tm/tm.sum(axis=1).reshape(-1,1)).cumsum(axis=1)    #Get probabilities


#Let run


dof = t.fit(data.log_returns.values)[0]     #guess for degrees of freedom
Wt = t.rvs(dof,loc=0.0,scale=1.0,size=(tsteps,npaths))
#Start the sim
for i in range(tsteps):
    sigma,current_levels = np.zeros(npaths),np.zeros(npaths)
    for j in range(3):
        ind = np.where(levels==j)[0]
        sigma[ind] = np.random.choice(vdict[str(j)],size=len(ind))
        current_levels[ind] = j
    Wt[i]*=sigma
    levels=current_levels
#Now to get the prices from the predicted returns
returns = np.vstack((np.ones(npaths),np.exp(Wt)))
s0 = data.Close.iloc[-1]
prices = s0*returns.cumprod(axis=0)
plt.plot(prices,color='b',linewidth=.1,alpha=.8);
