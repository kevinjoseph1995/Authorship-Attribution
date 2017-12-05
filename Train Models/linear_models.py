import pickle
from sklearn import  linear_model,neighbors
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler 



feature_vectors=pickle.load( open( "training_features.p", "rb" ) )
X=np.zeros([50*50,16])
Y=np.zeros(50*50)

index=0
author_index=0
author_index_dict={}
for author in feature_vectors:
    author_index_dict[author]=author_index
    for vector in feature_vectors[author]:
        X[index][:]=vector/np.linalg.norm(vector)
        Y[index]=author_index
        index+=1
    author_index+=1

random_indices=  np.random.permutation(X.shape[0])   
X=X[random_indices][:] 
Y=Y[random_indices]
n_samples = len(X)
X_train = X[:int(.90 * n_samples)]
y_train = Y[:int(.90 * n_samples)]
X_val = X[int(.90 * n_samples):]
y_val = Y[int(.90 * n_samples):]

scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# applying same transformation to test data
X_val = scaler.transform(X_val) 

<<<<<<< HEAD
knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=500)
print('LogisticRegression validation score: %f'
      % logistic.fit(X_train, y_train).score(X_val, y_val))   
print('KNN validation score: %f' % knn.fit(X_train, y_train).score(X_val, y_val))

clf = MLPClassifier(solver='lbfgs', alpha=0.1,hidden_layer_sizes=(100), random_state=1,max_iter=1000)
=======
#knn = neighbors.KNeighborsClassifier()
#logistic = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=1000)
#print('LogisticRegression validation score: %f'
#      % logistic.fit(X_train, y_train).score(X_val, y_val)) 
#print('LogisticRegression training score: %f'
#      % logistic.score(X_train, y_train))   
#print('KNN validation score: %f' % knn.fit(X_train, y_train).score(X_val, y_val))
#print('KNN training score: %f' % knn.score(X_train, y_train))
clf = MLPClassifier(solver='adam', alpha=0.1,hidden_layer_sizes=(50), random_state=1,max_iter=1000)
>>>>>>> 00683f5cb4014428bc7bbd5943a50d42637dd008
print('NN validation score: %f' % clf.fit(X_train, y_train).score(X_val, y_val))
print('NN training score: %f' % clf.score(X_train, y_train))
test_feature_vectors=pickle.load( open( "test_features.p", "rb" ) )
X_test=np.zeros([50*50,16])
Y_test=np.zeros(50*50)
index=0
author_index=0
author_index_dict={}
for author in test_feature_vectors:
    author_index_dict[author]=author_index
    for vector in test_feature_vectors[author]:
        X_test[index][:]=vector/np.linalg.norm(vector)
        Y_test[index]=author_index
        index+=1
    author_index+=1
X_test = scaler.transform(X_test) 
print ('NN test score: %f' % clf.score(X_test,Y_test))
#print('LogisticRegression test score: %f'
#      % logistic.score(X_test, Y_test))   
#print('KNN test score: %f' % knn.score(X_test, Y_test))
