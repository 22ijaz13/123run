import pandas as pd
import matplotlib.pyplot as plt


x=[2,3,4,5,7]
y=[6,9,12,16,19]

mean_x=sum(x)/len(x)
mean_y=sum(y)/len(y)

numer=0
denom=0

n=len(x)
for i in range(n):
    numer+=(x[i]-mean_x)*(y[i]-mean_y)
    denom+=(x[i]-mean_x)*(x[i]-mean_x)

m=numer/denom
b=mean_y-(m*mean_x)

k=int(input("Enter x value: "))
def prediction(a,bi,c):
   y_pred=a*c+bi
   return y_pred

print("Predicted y value: ",prediction(m,b,k))

plt.scatter(x,y,label='Data Points')
plt.plot(x,[prediction(m,b,i) for i in x], color='red', label='Linear Regression line')
plt.scatter(k,prediction(m,b,k),color='green',label="Predicted point")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()

