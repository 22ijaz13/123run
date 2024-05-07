import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
data=pd.read_csv("housing.csv")

data.dropna(inplace=True)

print(data.describe())

data.boxplot(column='total_bedrooms')
plt.show()

column_data=data['median_income']
q1=column_data.quantile(0.25)
q3=column_data.quantile(0.75)
iqr=q3-q1
lb=q1-1.5*iqr
ub=q1+1.5*iqr
outliers=column_data[(column_data<lb)|(column_data>ub)]
print(len(column_data))
column_data=column_data.drop(outliers.index)
print(len(column_data))

plt.scatter(data['total_rooms'],data['population'])
plt.show()

ndata=data.select_dtypes(include="number")
corr=ndata.corr()

print(corr)
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()




#REGRESSION
#Linear Regression

X=data[['total_bedrooms']]
y=data['population']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

r2=r2_score(y_test,y_pred)
print("R2 score:",r2)

plt.scatter(X_test_scaled,y_test,color='blue',label="actual points")
plt.plot(X_test_scaled,y_pred,color='red',label="regression line")
plt.legend()
plt.show()



X=data[['total_bedrooms','total_rooms']]
y=data['population']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

r2=r2_score(y_test,y_pred)
print("R2 score:",r2)

plt.scatter(X_test_scaled['total_bedrooms'],y_test,color='blue',label="actual points")
plt.plot(X_test_scaled["total_bedrooms"],y_pred,color='red',label="regression line 1")
plt.scatter(X_test_scaled['total_rooms'],y_test,color='blue',label="actual points")
plt.plot(X_test_scaled["total_rooms"],y_pred,color='black',label="regression line 2")
plt.legend()
plt.show()



#Polynomial regression

X=data[['total_bedrooms']]
y=data['population']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

scaler=MinMaxScaler()
#X_train_scaled=scaler.fit_transform(X_train)
#X_test_scaled=scaler.transform(X_test)
poly_features=PolynomialFeatures(degree=2)
X_train_poly=poly_features.fit_transform(X_train)
X_test_poly=poly_features.transform(X_test)
model1=LinearRegression()
model1.fit(X_train_poly,y_train)
yp=model1.predict(X_test_poly)
r2=r2_score(y_test,yp)
print("R2 score:",r2)
plt.scatter(X_test,y_test,color='blue',label="actual points")
plt.plot(X_test,yp,color='red',label="regression line")
plt.legend()
plt.show()


#KNN
x = [4, 5, 10, 4, 3, 11, 14 , 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

plt.scatter(x, y, c=classes)
plt.show()


data=list(zip(x,y))
k=3
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(data,classes)

new_x=8
new_y=21
new_point=[(new_x,new_y)]
prediction=knn.predict(new_point)

plt.scatter(x+[new_x],y+[new_y],c=classes+[prediction[0]])
plt.show()



df=pd.read_csv("Iris.csv")
X=df.drop(columns=['Species'])
y=df['Species']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
pca_df=pd.DataFrame(data=X_pca,columns=['pc1','pc2'])
print(pca_df)

plt.scatter(X_pca[:,0],X_pca[:,1],color='blue')
plt.title("PCA")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


df=pd.read_csv("Iris.csv")
X=df.drop(columns=['Species'])
y=df['Species']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)
y_p=clf.predict(X_test)

plt.figure(figsize=(15,10))
tree.plot_tree(clf)
plt.show()

a=accuracy_score(y_test,y_p)
print("Accuracy score:", a)

print("Classification Report:")
print(classification_report(y_test,y_p))

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_p))
