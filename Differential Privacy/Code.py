#Libraries Import
import numpy as np; import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from math import e
import math
# Importing the data 
nfdata = pd.read_csv("C:\\Users\\Akshey12\\Desktop\\Python\\Differential Privacy\\Netdata.csv")
#Header of the columns
header = list(nfdata['User'])
#Initialization of values and constants
ep = 0.001                                 #float(input("Enter the  Ɛ (epsilon value):"))  
k =  0.5                                   #int(input("Enter the value of k :"))          
si = 0.8                                   #float(input("Enter the Si value:"))           
w =  0.1                                  #float(input("Enter the w value:"))       

#Please input the user id and movie id for which Predicted Rating is Required     
user = int(input("Enter the user id :"))
movie = int(input("Enter the movie id:"))
c1 =[]
c0 =[]
# Extracting the index corresponding to user
usi = int(nfdata[nfdata['User']== user].index[0])
#Remove Users Column as it will effect our calculations
data = nfdata.iloc[0:,1:]
#Total Movies & Users Count
mcount = len(list(data))
ucount = len(header)

#Neighborhood-Based Methods: k Nearest Neighbors

""""Two stages are involved in neighborhood-based methods: the Neighbor Selection
and the Rating Prediction.In the Neighbor Selection stage, the similarity between any 
two users or any two items is estimated, and corresponds to the user-based methods and the item-based methods.
Various measurement metrics have been proposed to compute the similarity.Two
of the most popular ones are the Pearson Correlation Coefficient (PCC) and
Cosine-based Similarity (COS) . Neighbors are then selected according to the
similarity.In the Rating Prediction Stage, for any item ti, all ratings on ti by users in Nk.ua/
will be aggregated into the predicted ratingbrai. Specifically, the prediction ofbrai is
calculated as a weighted sum of the neighbors’ ratings on item ti. Accordingly, the
determination of a suitable set of weights becomes an essential problem because
most work relies on the similarity between users or items to determine the weight.
""""
# Creation of Cosine Similiarity Matrix
cos = cosine_similarity(data)

#Converting the data into dataframe
Sim = pd.DataFrame(data=cos[0:,0:],  
         index=cos[0:,0],    
         columns=cos[0,0:]) 
		 
#Adding the column headers
Frame=pd.DataFrame(Sim.values, columns = header)

#Sorting the column containing the missing value
SortFrame = Frame.sort_values((user), ascending = [0])

""""
Private Neighbor Collaborative Filtering Algorithm

For the privacy preserving issue in the context of neighborhood-based CF methods,
the preserving targets differ between item-based methods and user-based methods
due to the different perspectives regarding definition of similarity.
In user-based methods, what an adversary can infer from the user
similarity matrix is the item rated by the active user. The preserving objective is
then to hide the historically rated items. The proposed PNCF algorithm can deal
with both cases. To make it clear, Zhu et al. presented the PNCF algorithm from
the perspective of the item-based methods, and this can be applied to user-based
methods in a straightforward manner.The first
stage aims to identify the items of k nearest neighbor, and the second stage aims to
predict the rating by aggregating the ratings on those identified neighbor items. To
resist a KNN attack, the neighbor information in both stages should be preserved.""""

# Storing the value greater than si-w in different array 
for i, row in SortFrame.iterrows():
    if row[user] > (si-w):
        c1.append(SortFrame.loc[i,:])
    else :
        c0.append(SortFrame.loc[i,:])

""""Item candidate list I is divided into two sets: C1 and C0. Set C1 consists of items
whose similarities are larger than the truncated similarity and C0
consists of the remaining items in I."""
		
C1 = np.asarray(c1)
C0 = np.asarray(c0)

"""""The first operation is to define a new Recommendation-Aware Sensitivity to decrease the
noise, and the second operation is to provide a new recommender mechanism to
enhance the accuracy. Both are integrated to form the proposed PNCF algorithm,
which consequently obtains a better trade-off between privacy and utility.

Recommendation-Aware Sensitivity
This section presents Recommendation-Aware Sensitivity based on the notion of
Local Sensitivity to reduce the magnitude of noise introduced for privacy-preserving
purposes. Recommendation-Aware Sensitivity for score function q, RS.i; j/ is measured
by the maximal change in similarity of two items when removing a user’s
rating record."""

#Defining the new array for Recommendation-Aware Sensitivity 
RS = np.zeros(shape=(ucount,ucount))
S1 =[]
# Storing values in RAS MATRIX
for i , rows in data.iterrows():
    for j , rows in data.iterrows():
        for k in range(0,mcount):
            S1.append(cosine_similarity(data.iloc[i:i+1, data.columns != (data.columns[k])],data.iloc[j:j+1, data.columns != (data.columns[k])]))
        R = min(S1)
        RS[i][j]= R
        S1 = []
#Conversion of Array to Dataframe
Rs= pd.DataFrame(data=RS[0:,0:],  
         index=RS[0:,0],    
         columns=RS[0,0:]) 
		 
#Adding headers of RAS Matrix
RS_Frame=pd.DataFrame(Rs.values, columns = header)

""""This algorithm uses a new notion of truncated similarity (S Hat)as the
score function to enhance the quality of selected neighbors. The truncated notion
was first mentioned in Bhaskar et al.’s work [24], in which they used truncated
frequency to decrease the computational complexity. The same notion can be used
in our score function q to find those high quality neighbors. Specifically, for each
item in the candidate list I, if its similarity s(i,j) is smaller than s(i,.)-w then s(i,j)
is truncated to s(i,.)-w where w is a truncated parameter in the score function;
otherwise it is still preserved as s(i,j).""""

#Initializing S Hat
S_ht = np.zeros(shape=(ucount,ucount))  

#Conversion of Array to Dataframe
Sh= pd.DataFrame(data=S_ht[0:,0:],  
         index=S_ht[0:,0],    
         columns=S_ht[0,0:])
#Storing values in S^ Matrix
for i , rows in Frame.iterrows():
    for j , rows in Frame.iterrows():
        if Frame.iloc[i,j]> (si-w):
            Sh.iloc[i,j] = Frame.iloc[i,j]
        else :
            Sh.iloc[i,j] = (si-w)
Shat_Frame=pd.DataFrame(Sh.values, columns = header)

""""Implementation of Algorithm : Private Neighbor Selection""""
#Initialization of Array for Calculations
Num = np.zeros(shape=(ucount,1)) 

#Conversion of Array to Dataframe
Nu_pr= pd.DataFrame(data=Num[0:,0:],  
          index=Num[0:,0],    
          columns=Num[0,0:]) 
		  
#Dropping a row
Frame.drop(Frame.index[[usi]])
Shat_Frame.drop(Shat_Frame.index[[usi]])
RS_Frame.drop(RS_Frame.index[[usi]])


#Calculating Numerator Part for the Probability
for i , rows in Shat_Frame.iterrows():
    Nu_pr.iloc[i,0] = (e**(ep * Shat_Frame.iloc[i,usi])/(4*k*RS_Frame.iloc[i,usi]))
#Sum 
Total =  Nu_pr.iloc[:,0].sum()
##Initialization of Array for Calculating Probability
Prop = np.zeros(shape=(ucount,1)) 

#Conversion of Array to Dataframe
Prb = pd.DataFrame(data=Prop[0:,0:],  
          index=Prop[0:,0],    
          columns=Prop[0,0:]) 
#Storing values in Probability Matrix
for i in range(0,ucount):
    Prb.iloc[i,0] = ((Nu_pr.iloc[i,0])/Total)

##Initialization of Arrays for Final Ratings
fr1 = np.zeros(shape=(ucount,1))
fr2 = np.zeros(shape=(ucount,1))
F_rate1 = pd.DataFrame(data=fr1[0:,0:],  
          index=fr1[0:,0],    
          columns=fr1[0,0:]) 
F_rate2 = pd.DataFrame(data=fr2[0:,0:],  
          index=fr2[0:,0],    
          columns=fr2[0,0:]) 
#Calculation of Ratings
for i in range(0,ucount):
    F_rate1.iloc[i,0] = nfdata.iloc[i,nfdata.columns.get_loc(str(movie))]* Prb.iloc[i,0]*Frame.iloc[i,usi]
    F_rate2.iloc[i,0] = Prb.iloc[i,0]*Frame.iloc[i,usi]

F_rate1.drop(F_rate1.index[[usi]])
# Final Rating Output After Applying Differential Privacy
Rating = (F_rate1.iloc[:,0].sum()-F_rate1.iloc[usi,0])/(F_rate2.iloc[:,0].sum())
if Rating-(math.floor(Rating)) > 0.5:
    print(math.ceil(Rating))
else :
    print(math.floor(Rating))
