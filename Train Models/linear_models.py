import pickle
from sklearn import  linear_model,neighbors
from sklearn.neural_network import MLPClassifier
import numpy as np




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
X_test = X[int(.90 * n_samples):]
y_test = Y[int(.90 * n_samples):]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=100)
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))   
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(2000), random_state=1)
print('NN score: %f' % clf.fit(X_train, y_train).score(X_test, y_test))

