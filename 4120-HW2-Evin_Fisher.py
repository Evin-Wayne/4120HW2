#!/usr/bin/env python
# coding: utf-8

# In[39]:


# Import Libraries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets.samples_generator import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize dataset
X, yTrue = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


# In[40]:


# Initialize the model and graph
m1 = KMeans()
graph = KElbowVisualizer(m1, k=(1,12))

#Show K-elbow Graph with data
graph.fit(X)
graph.show()
print("The best K is 4")


# In[41]:


# Accuracy for best K
m2 = KMeans(4)
m2.fit(X)
yPrediction = m2.predict(X)
accuracy_score(yTrue, yPrediction)


# In[42]:


# Confusion matrix
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
matrix = confusion_matrix(yTrue, yPrediction)
sns.heatmap(matrix, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True')
plt.ylabel('Predicted')


# In[ ]:





# In[ ]:




