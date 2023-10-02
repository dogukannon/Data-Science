import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset("titanic")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df.head()


df["sex"].value_counts().plot(kind="bar")
plt.show()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

############
#plot
############

x = np.array([1, 10])
y = np.array([1, 100])
plt.plot(x, y)
plt.show()
plt.plot(x, y, "o")
plt.show()

x = np.array([0, 2, 4, 6, 8, 10])
y = np.array([0, 4, 16, 36, 64, 100])
plt.plot(x, y)
plt.show()
plt.plot(x, y, "o")
plt.show()


############
#marker
############
y = np.array([1, 4, 6, 3, 12, 8])
plt.plot(y, marker="*")
plt.show()

############
#line
############
y = np.array([1, 4, 6, 3, 12, 8])
plt.plot(y, linestyle="dashed", color="r")
plt.show()


############
#multiple lines
############
y = np.array([1, 4, 6, 3, 12, 8])
x = np.array([2, 3, 8, 0, 14, 19])
plt.plot(y, linestyle="dashed", color="r")
plt.plot(x, linestyle="dotted", color="b")
plt.show()

############
#labels
############
y = np.array([1, 4, 6, 3, 12, 8])
x = np.array([2, 3, 8, 0, 14, 19])
plt.plot(x, y, "o")
plt.title("x versus y")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

############
#subplots
############
y = np.array([1, 4, 6, 3, 12, 8])
x = np.array([2, 3, 8, 0, 14, 19])
plt.subplot(1,2,1)
plt.title("1")
plt.plot(x,y)

z = np.array([3, 5, 7, 8, 15, 16])
v = np.array([5, 13, 23, 6, 15, 2])
plt.subplot(1,2,2)
plt.title("2")
plt.plot(z,v)
plt.show()


############
#seaborn
############
df1 = sns.load_dataset("tips")
df1["sex"].value_counts()
sns.countplot(x=df1["sex"],data=df1)
sns.boxplot(x=df1["total_bill"])
sns.scatterplot(x=df1["tip"],y=df1["total_bill"],
                hue=df1["smoker"], data=df1)