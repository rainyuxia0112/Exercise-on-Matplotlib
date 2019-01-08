from sklearn.datasets import load_iris
iris = load_iris()

import matplotlib.pyplot as plt
import pandas as pd
# choosing x-data from iris
arr=iris['data']

# using feature-names as colunm name
names=iris['feature_names']
arr=arr.T

# zip two list
new_data=list(zip(names,arr))
dic=dict(new_data)
# change dic into dataframe
df=pd.DataFrame(dic)
# using df first colunm to draw histgram
df['sepal length (cm)'].plot(kind='hist',normed=True,bins=30,color='r')
plt.show()
# save as a pdf
plt.savefig('iris.pdf')
df['sepal length (cm)'].plot(kind='hist',normed=True,bins=30,cumulative=True)
plt.show()

# plot 4 figures into different plot
df.plot(kind='hist',bins=30,range=(0,10),normed=True,cumulative=True,subplots=True)
plt.title('histgram on iris data')
plt.ylabel('cumulative')
plt.xlabel('iris data')
plt.show()
plt.savefig('iris.pdf')

# plot box using iris data
df.plot(kind='box',subplots=True)
plt.show()


# statistical  describe
df['petal width (cm)'].describe()




