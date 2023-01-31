import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split


path = r"C:\Users\Admin\Desktop\Python_ML_class\Decision_tree\diabetes.csv"
df1=pd.read_csv(path)

print(df1.describe())

print(df1.value_counts("Outcome"))

sns.pairplot(df1, kind="scatter", hue='Outcome')
# plt.show()


# Drop the outcome:
    
df_corr=df1.drop(["Outcome"],axis=1)

print(df_corr)

#Building heat map

corr_matrix=df_corr.corr()

print(corr_matrix)

mask=np.zeros_like(corr_matrix)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr_matrix,mask=mask,square=False)


fig, axs=plt.subplots(4,2,figsize=(15,15))

sns.histplot(data=df1,x="Pregnancies",hue="Outcome",kde=True,color="skyblue",ax=axs[0,0])
sns.histplot(data=df1,x="Glucose",hue="Outcome",kde=True,color="skyblue",ax=axs[0,1])
sns.histplot(data=df1, x="BloodPressure", hue="Outcome", kde=True, color="skyblue", ax=axs[1, 0])
sns.histplot(data=df1, x="SkinThickness", hue="Outcome", kde=True, color="skyblue", ax=axs[1, 1])
sns.histplot(data=df1, x="Insulin", hue="Outcome", kde=True, color="skyblue", ax=axs[2, 0])
sns.histplot(data=df1, x="BMI", hue="Outcome", kde=True, color="skyblue", ax=axs[2, 1])
sns.histplot(data=df1, x="DiabetesPedigreeFunction", hue="Outcome", kde=True, color="skyblue", ax=axs[3, 0])
sns.histplot(data=df1, x="Age", hue="Outcome", kde=False, color="skyblue", ax=axs[3, 1])
plt.show()

fig, hst=plt.subplots(4,2,figsize=(15,15))

sns.boxplot(x=df1["Outcome"],y=df1["Pregnancies"],ax=hst[0,0])
sns.boxplot(x=df1["Outcome"], y=df1["Glucose"], ax=hst[0, 1])
sns.boxplot(x=df1["Outcome"], y=df1["BloodPressure"], ax=hst[1, 0])
sns.boxplot(x=df1["Outcome"], y=df1["SkinThickness"],ax=hst[1, 1])
sns.boxplot(x=df1["Outcome"], y=df1["Insulin"], ax=hst[2, 0])
sns.boxplot(x=df1["Outcome"], y=df1["BMI"], ax=hst[2, 1])
sns.boxplot(x=df1["Outcome"], y=df1["DiabetesPedigreeFunction"], ax=hst[3, 0])
sns.boxplot(x=df1["Outcome"], y=df1["Age"], ax=hst[3, 1])
plt.show()

x=df1.drop(["Outcome"], axis=1)
y=df1["Outcome"]

x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=42)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
# train_test_split()
# print(tree)
print("accuracy of training set: %f" %tree.score(x_train,y_train))
print("accuracy on test set: %f" %tree.score(x_test,y_test))

tree.feature_importances_

## now the model is in overfit. Required to reduce the overfitness in the model

plt.plot(tree.feature_importances_,'o')
plt.xticks(range(x.shape[1]),x.columns,rotation=90)
plt.ylim(0,1)


tree = DecisionTreeClassifier(max_depth=3)
tree.fit(x_train,y_train)
print("accuracy on training set: %f" % tree.score(x_train, y_train))
print("accuracy on test set: %f" % tree.score(x_test, y_test))
tree.feature_importances_


plt.plot(tree.feature_importances_,'o')
plt.xticks(range(x.shape[1]),x.columns,rotation=90)
plt.ylim(0,1)

#To build a tree diagram

from sklearn.tree import export_graphviz
import graphviz
import pydot

#To create a tree

export_graphviz(tree,out_file="My_tree.dot",class_names=["Diabetes","No-Diabetecs"],
                feature_names=x.columns,filled=False,impurity=True)

# tree Visualization
with open("My_tree.dot") as f:
    dot_graph=f.read()
graphviz.Source(dot_graph)    

# Use dot file to create a graph:
(graph, ) = pydot.graph_from_dot_file('My_tree.dot')
# Write graph to a png file:
tree_png = graph.write_png('My_tree.png')









