from joblib import load
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class Classifier:
    def __init__(self):
        self.svm = load('models/SequenceVectorMachine.joblib')
        self.dt = load('models/DecisionTree.joblib')
        self.knn = load('models/KNN.joblib')
        self.rf = load('models/RandomForest.joblib')
        pass
    
    def getSVM(self):
        return self.svm
    
    def getRF(self):
        return self.rf
    
    def getDT(self):
        return self.dt
    
    def getKNN(self):
        return self.knn
    
    def avg(self,a):
        sum = 0
        for i in a:
            sum = sum + i
        return sum/len(a)
    
    def predictOutput(self,model,X,y):
        y_pred = model.predict(X)
        acc_scr = accuracy_score(y, y_pred)
        prf = precision_recall_fscore_support(y, y_pred)
        precision = self.avg(prf[0])
        recall = self.avg(prf[1])
        fscore = self.avg(prf[2])
        
        return acc_scr,precision,recall,fscore
        