import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import pickle

# In[2]:


dataset = pd.read_csv('blindchess_dataset.csv')



# # Step 3: Data Preprocessing

# In[13]:
new_dataset = dataset
new_dataset[["Moves", "Rating"]] = new_dataset[["Moves", "Rating"]].replace(0, np.NaN) 
new_dataset.isnull().sum()
new_dataset["Moves"].fillna(new_dataset["Moves"].mean(), inplace = True)
new_dataset["Rating"].fillna(new_dataset["Rating"].mean(), inplace = True)
convert_dict = {'Moves': int,
                'Rating': int
               }
  
new_dataset = new_dataset.astype(convert_dict)

dataset_X = dataset.iloc[:,[0]].values
dataset_Y = dataset.iloc[:,1].values

# print(dataset_X)

# In[14]:


dataset_X


# In[15]:


#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range = (0,1))
#dataset_scaled = sc.fit_transform(dataset_X)


# In[16]:


dataset_scaled =  new_dataset

# print(dataset_scaled)

# In[17]:


X = dataset_X
Y = dataset_Y


# In[18]:


X


# In[19]:


Y


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.50, random_state = 42, stratify = dataset['Rating'] )


# # Step 4: Data Modelling

# In[25]:


from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)


# In[26]:


svc.score(X_test, Y_test)


# In[27]:


Y_pred = svc.predict(X_test)

# print("X_test")
# print(X_test)
# print("Y_pred")
# print(Y_pred)



pickle.dump(svc, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[25]]))
#print(model.predict(sc.transform(np.array([[86, 66, 26.6, 31]]))))


