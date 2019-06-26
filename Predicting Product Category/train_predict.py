#!/usr/bin/env python
# coding: utf-8

# # Import Libraries



import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# # Read in Data




df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')


# # Create dictionary for categorical variables

# In[3]:


def col_dict(df_input, col ):
    temp_list = list(df[col].unique())
    temp_dict = {}
    for item in temp_list:
        temp_dict[item] = temp_list.index(item)
    print(col+'_dict Length: ',len(temp_dict))
    return temp_dict

Vendor_Code_dict = col_dict(df, col = 'Vendor_Code')
Product_Category_dict = col_dict(df, col = 'Product_Category')
GL_Code_dict = col_dict(df, col = 'GL_Code')


# # Modify the Description Column




# Item Description is the key factor in determining the Category of the product
# So, the keyword are extracted for further analysis
des_list = df['Item_Description'].tolist()
des_words = [a.split() for a in des_list]

des_words = list(des_words)
words = []
for a in range(len(des_words)):
    words = words + des_words[a]

def rchar(s, c) : 
    for item in c:
        counts = s.count(item) 
        s = list(s) 
        while counts :  
            s.remove(item) 
            counts -= 1 
        s = '' . join(s)
    return s 
  

b = [rchar(item, ['\\', '/', '(', ')', '-', '.', ' ']) for item in words]
c = b.copy()
for item in b:
    if len(item) < 3:
        c.remove(item)      

words = [a.lower() for a in c]

remove = ['of', 'on', 'with', 'and']
count = 0
for a in words:
    if a.lower() in remove:
        words.remove(a)
        count += 1

words = pd.Series(words)
words_count = words.value_counts()

words_list_unique = words_count.index
type(list(words_list_unique))
print('words_list_unique length: ',len(words_list_unique))


# # Prepare data for Learning




def modi(df_input, type_input = 'predict'):
    df = df_input.copy()
    item_des_series = list(df['Item_Description'])
    wordscount_df = pd.DataFrame()
    for a in words_list_unique:
        list_1 = []
        for item in item_des_series:
            if a in item:
                list_1.append(int(1))
            else:
                list_1.append(int(0))
        wordscount_df[a] = list_1

    wordscount_df['Inv_Id'] = df['Inv_Id']
    all_df = pd.merge(df,wordscount_df, how = 'inner', on = 'Inv_Id')

    for i in range(all_df.shape[0]):
        if all_df.loc[i,'Vendor_Code'] in Vendor_Code_dict:
            all_df.loc[i,'Vendor_Code'] = Vendor_Code_dict[all_df.loc[i,'Vendor_Code']]
        else:
            Vendor_Code_dict[all_df.loc[i,'Vendor_Code']] = len(Vendor_Code_dict)
            all_df.loc[i,'Vendor_Code'] = Vendor_Code_dict[all_df.loc[i,'Vendor_Code']]
                
    for i in range(all_df.shape[0]):
        if all_df.loc[i,'GL_Code'] in GL_Code_dict:
            all_df.loc[i,'GL_Code'] = GL_Code_dict[all_df.loc[i,'GL_Code']]
        else:
            GL_Code_dict[all_df.loc[i,'GL_Code']] = len(GL_Code_dict)
            all_df.loc[i,'GL_Code'] = GL_Code_dict[all_df.loc[i,'GL_Code']]
            
    if type_input != 'predict':
        for i in range(all_df.shape[0]):
            if all_df.loc[i,'Product_Category'] in Product_Category_dict:
                all_df.loc[i,'Product_Category'] = Product_Category_dict[all_df.loc[i,'Product_Category']]
            else:
                Product_Category_dict[all_df.loc[i,'Product_Category']] = len(Product_Category_dict)
                all_df.loc[i,'Product_Category'] = Product_Category_dict[all_df.loc[i,'Product_Category']]
#-----------------------------------------------------------------------------------------------------------#    
    col = list(all_df.columns)

    col.remove('Inv_Id')
    col.remove('Item_Description')
    if type_input == 'predict':
        X = all_df[col] .values  #.astype(float)
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        return X
#        y = all_df['Product_Category'].values
    else:
        col.remove('Product_Category')
        X = all_df[col] .values  #.astype(float)
        y = all_df['Product_Category'].values
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float)) 
        return X,y


# # Build the model




X_predict = modi(test_df, type_input = 'predict')
X_train,y_train = modi(df, type_input = 'train')
#-------------------------------------------------------
k = 3
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
#----------------------------------------------------
y_predict = neigh.predict(X_predict)
#----------------------------------------------------------
Answer = []
Product_Category_dict_swap = dict([(value, key) for key, value in Product_Category_dict.items()])
for item in y_predict:
    Answer.append(Product_Category_dict_swap[item])

Answer = pd.Series(Answer)
test_df['Product_Category'] = Answer
submit = test_df[['Inv_Id', 'Product_Category']]





submit.to_csv('Answer.csv', index=False)

