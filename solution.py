
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model



# In[29]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[30]:


print(train.shape)


# In[31]:


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']=(10,6)
print(train['SalePrice'].describe())


# In[32]:


print(train['SalePrice'].skew())


# In[33]:


plt.hist(train['SalePrice'],color='red')
plt.show()


# In[34]:


target = np.log(train['SalePrice'])
print(target.skew())


# In[35]:


numeric_features = train.select_dtypes(include =[np.number])
corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending = False)[:5])
print(corr['SalePrice'].sort_values(ascending =False)[-5:])


# In[42]:


train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'],y=np.log(train['SalePrice']))
plt.show()


# In[46]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


# In[47]:


categorical = train.select_dtypes(exclude =[np.number])
print(categorical.describe())


# In[49]:


print(train['Street'].value_counts())


# In[52]:


train['new_street']=pd.get_dummies(train['Street'],drop_first=True)
test['new_street']=pd.get_dummies(train['Street'],drop_first=True)
print(train['new_street'].value_counts())


# In[71]:


def encode(x): return 1 if x=='Partial' else 0
train['enc_saleCondition']= train['SaleCondition'].apply(encode)
test['enc_saleCondition'] = test['SaleCondition'].apply(encode)
condition_pivot = train.pivot_table(values ='SalePrice',index='enc_saleCondition',aggfunc = np.median)
condition_pivot.plot(kind='bar',color ='red')

#plt.show()


# In[73]:


data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print(data.isnull().sum())


# In[74]:


y = np.log(train['SalePrice'])
x = data.drop(['SalePrice','Id'],axis = 1)


# In[75]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=.33)


# In[76]:


model = linear_model.LinearRegression()
model.fit(x_train,y_train)


# In[77]:


print(model.score(x_test,y_test))


# In[80]:


predictions = model.predict(x_test)
print(mean_squared_error(y_test,predictions))


# In[81]:


#for i in range(-2,3):
 #   alpha = 10**i;
  #  rm = linear_model.Ridge(alpha=alpha)
   # ridge_model = rm.fit(x_train,y_train)
    #ridge_pred = ridge_model.predict(x_test)
    


# In[87]:


submission = pd.DataFrame()
submission['Id']=test['Id']
fe = test.select_dtypes(include=[np.number]).drop(['Id'],axis=1).interpolate()
new_predictions = model.predict(fe)
final_predictions = np.exp(new_predictions)
submission['SalePrice']=final_predictions
print(submission.head())


# In[88]:


submission.to_csv("submit.csv",index=False)

