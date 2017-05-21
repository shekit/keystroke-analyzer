
# coding: utf-8

# In[1]:

import numpy as np
import keras


# In[2]:

from keras.models import Sequential
from keras.layers import Dense


# In[13]:

def load_file(filename):
    
    #return np.array(np.load(open(filename, 'r')), dtype='float32')
    
    csv = np.array(np.genfromtxt(filename, delimiter=","), dtype='float64')
    return csv


# In[4]:

# split data before scaling

def split_train_val(percent, data):
    num = int(percent/100*len(data))
    train, test = data[:num,:], data[num:,:]
    return train, test


# In[5]:

def combine(one,two):
    return np.concatenate((one,two), axis=0)


# In[6]:

def create_binary_labels(num):
    return np.array([0]*num + [1]*num)


# In[7]:

# find only for training data

def colmin(data):
    mins = []
    
    for i in range(len(data[0])):
        mins.append(min(data[:,i]))
        
    return mins


# In[8]:

# find only for training data

def colmax(data):
    maxs = []
    
    for i in range(len(data[0])):
        maxs.append(max(data[:,i]))
    
    return maxs


# In[9]:

# use values from training data to scale everything, split first then scale

def applyMinMax(data, mins, maxs):
    
    for i in range(len(data)):
        for t in range(len(data[0])):
            data[i][t] = (data[i][t]-mins[t])/(maxs[t]-mins[t])
            
    return data


# In[14]:

leon = load_file('leon10.csv')
chino = load_file('chino10.csv')


# In[ ]:




# In[18]:

leon_train, leon_test = split_train_val(80,leon)
chino_train, chino_test = split_train_val(80,chino)


# In[20]:

print(chino_train.shape)


# In[21]:

train = combine(leon_train, chino_train)
test = combine(leon_test, chino_test)


# In[22]:

print(train.shape)


# In[23]:

y_train = create_binary_labels(leon_train.shape[0])
y_test = create_binary_labels(leon_test.shape[0])


# In[26]:


# In[27]:

minimum = colmin(train)


# In[28]:

print(minimum)


# In[29]:

maximum = colmax(train)


# In[30]:

print(maximum)


# In[31]:

normalized_train = applyMinMax(train, minimum, maximum)


# In[33]:

normalized_test = applyMinMax(test, minimum, maximum)


# In[ ]:




# In[ ]:




# In[ ]:




# In[4]:

model = Sequential()

model.add(Dense(128, input_shape=(4,), activation='tanh'))
model.add(Dense(1, activation='sigmoid'))


# In[5]:

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# In[ ]:

model.fit(normalized_train, y_train, verbose=1, batch_size=128, epochs=20, validation_data=(normalized_test, y_test))


# In[ ]:

model.save('keyboard.h5')

