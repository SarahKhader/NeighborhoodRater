
import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as pw

# The following data should be provided based on the user to whom we are looking for an apartement
average_distence_to_center_from_work = 2
park = 1 # the user is interested of having a Park around
school = 1 # the user has kids in school age
kita =0 # user does not have kids who need kita
night_clubs = 0 # user does not party alot
dog_area =1 # user has a dog
average_rent = 950 # Around third of users salary

neighborhoods = pd.read_csv('Neighborhood_DataSet.csv')
neighborhoods['rating']=0
newUser = np.array([0,average_distence_to_center_from_work,park,school,kita,night_clubs,dog_area,average_rent], dtype=float).reshape(1,-1)


for i in range(len(neighborhoods)) : 
  neighborhoods.loc[i,'rating']=pw.cosine_similarity(np.array(neighborhoods.loc[i])[1:].reshape(1,-1),newUser)[0][0]

print(neighborhoods.sort_values(['rating'],ascending=False))
