import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 


class DataSource:
    def __init__(self):
        self.data = pd.read_csv('data\data.csv')
        st_x= StandardScaler()  
        self.data_scaled = st_x.fit_transform(self.data) 
        self.data_scaled = pd.DataFrame(self.data_scaled)   
        self.X,self.y = self.data_scaled.iloc[:,0:55],self.data['Cover_Type']
        self.trainX,self.testX,self.trainY,self.testY = train_test_split(self.X,self.y, test_size=0.2, random_state=69, shuffle=True, stratify=None) 
    
    def getHead(self):
        return self.data.head(n=5)
    
    def getTrainData(self):
        return self.trainX,self.trainY
    
    def getTestData(self):
        return self.testX,self.testY
    
    def describeData(self):
        return self.data.describe()