import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
#np.array([[y]])
#y=np.reshape(y,(len(y),1))
y=pd.DataFrame(y)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(6.5)))


plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("truth or bluff (SVR) model")
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

