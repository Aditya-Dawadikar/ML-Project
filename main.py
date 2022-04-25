import streamlit as st

from data.DataSource import DataSource
from models.Classifier import Classifier

header = st.container()
body = st.container()

data_src = DataSource()
X_test,y_test = data_src.getTestData()

classifier = Classifier()

    

with header:
    header.title("ML mini project")
    header.subheader("Forest Cover Type Prediction using cartographic variables")
    header.image('./assets/forest.jfif')
    header.subheader("What is cartography?")
    header.write("Cartography is the study and practice of making and using maps. Combining science, aesthetics and technique, cartography builds on the premise that reality can be modeled in ways that communicate spatial information effectively.")
    header.write("[Dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction)")
    
with body:
    dataset_description = st.container()
    models_used = st.container()
    analysis_results = st.container()
    
    with dataset_description:
        dataset_description.title("Dataset description:")
        desc = data_src.describeData()
        dataset_description.write(desc)
        
        dataset_description.title("Sample Data:")
        dataset_description.write(data_src.getHead())
        
    with models_used:
        models_used.title("ML models:")
        
        model_1 = st.container()
        model_2 = st.container()
        model_3 = st.container()
        model_4 = st.container()
        
        with model_1:
            model_1.subheader("1. K Nearest Neighbors")
            model_1.markdown("``` classifier = KNeighborsClassifier(n_neighbors=10) ```")
            if model_1.button('Test KNN'):
                knn = classifier.getKNN()
                acc,p,r,f = classifier.predictOutput(knn,X_test,y_test)
                model_1.write("accuracy: "+str(acc))
                model_1.write("precision: "+str(p))
                model_1.write("recall: "+str(r))
                model_1.write("fscore: "+str(f))
                
        
        with model_2:
            model_2.subheader("2. Sequence Vector Machine")
            model_2.markdown("``` classifier = SVC(kernel='rbf', gamma=0.5, C=0.1) ```")
            if model_2.button('Test SVM'):
                svm = classifier.getSVM()
                acc,p,r,f = classifier.predictOutput(svm,X_test,y_test)
                model_2.write("accuracy: "+str(acc))
                model_2.write("precision: "+str(p))
                model_2.write("recall: "+str(r))
                model_2.write("fscore: "+str(f))
        
        with model_3:
            model_3.subheader("3. Decision Tree")
            model_3.markdown("``` classifier= DecisionTreeClassifier(criterion='entropy', random_state=4) ```")
            if model_3.button('Test DT'):
                dt = classifier.getDT()
                acc,p,r,f = classifier.predictOutput(dt,X_test,y_test)
                model_3.write("accuracy: "+str(acc))
                model_3.write("precision: "+str(p))
                model_3.write("recall: "+str(r))
                model_3.write("fscore: "+str(f))
        
        with model_4:
            model_4.subheader("4. Random Forest")
            model_4.markdown("``` classifier = RandomForestClassifier(n_neighbors=10) ```")
            if model_4.button('Test RF'):
                rf = classifier.getRF()
                acc,p,r,f = classifier.predictOutput(rf,X_test,y_test)
                model_4.write("accuracy: "+str(acc))
                model_4.write("precision: "+str(p))
                model_4.write("recall: "+str(r))
                model_4.write("fscore: "+str(f))
    
    with analysis_results:
        analysis_results.title("Analysis Results")
        analysis_results.subheader("Comparison Using Confusion Matrix")
        
        row_1 = st.container()
        row_2 = st.container()
        
        with row_1:
            col_1,col_2 = st.columns(2)
            
            with col_1:
                col_1.subheader("KNN classifier")
                col_1.image('./assets/knn.png')
                col_1.write('Accuracy: '+str(0.7658))
            with col_2:
                col_2.subheader("SVM classifier")
                col_2.image('./assets/svm.png')
                col_2.write('Accuracy: '+str(0.7678))
            
        with row_1:
            col_1,col_2 = st.columns(2)
            
            with col_1:
                col_1.subheader("Decision Tree classifier")
                col_1.image('./assets/dt.png')
                col_1.write('Accuracy: '+str(0.7939))
            with col_2:
                col_2.subheader("Random Forest classifier")
                col_2.image('./assets/rf.png')
                col_2.write('Accuracy: '+str(0.8683))