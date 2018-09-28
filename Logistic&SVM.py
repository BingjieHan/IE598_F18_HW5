import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from matplotlib.colors import ListedColormap

def accuracy_results(clf, X_train, y_train, X_test, y_test):
    cv_score = np.mean(cross_val_score(clf, X_train, y_train, cv=5 , scoring='accuracy'))
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    #train_score =  clf.score(X_train, y_train)
    #test_score = clf.score(X_test, y_test)
    train_score = accuracy_score(y_train,y_pred_train)
    test_score = accuracy_score(y_test,y_pred_test)
    scores = [cv_score, train_score, test_score]
    return (scores)

def plot_decision_regions(X, y, classifier, test_idx=None,  
                          resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')
        
#load data
df = pd.read_csv('wine.csv',header=0)
#print basic information of dataset
print(df.head(),'\n')
print(df.describe())

df.head().to_excel('1.xls')
df.describe().to_excel('2.xls')

#make the boxplot
array = df.values
plt.boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel(("Quartile Ranges"))
plt.savefig('boxplot.png')
plt.show()

#make the heatmap
corMat = pd.DataFrame(df.corr())
print(corMat)
_=sns.heatmap(df.corr(), square=True, cmap='RdYlGn',annot=True, annot_kws={'size':7})
plt.savefig('heatmap.png')

sns.pairplot(df,size=2.5)
plt.tight_layout()
plt.show()


#split the dataset
X=df.iloc[:,0:len(df.columns)-1].values
y=df.iloc[:,-1].values
#y=y.reshape((len(df),1))
sc_x=StandardScaler()
sc_x.fit(X)
X_std = sc_x.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, 
                                                    random_state=42)
print()

#boxplot after standardization
array=np.append(X_std,y,axis=1)
plt.boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel(("Quartile Ranges"))
plt.savefig('boxplot.png')
plt.show()


#logistic regression
log = LogisticRegression(multi_class='ovr')
log.fit(X_train,y_train)

#SVM
svm=SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

#PCA    
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
#print(pca.explained_variance_[np.cumsum(pca.explained_variance_ratio_)<0.8])
log_pca=LogisticRegression(multi_class='ovr')
log_pca.fit(X_train_pca, y_train)
svm_pca=SVC(kernel='linear',C=1.0)
svm_pca.fit(X_train_pca, y_train)

#LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)
log_lda=LogisticRegression(multi_class='ovr')
log_lda.fit(X_train_lda,y_train)
svm_lda=SVC(kernel='linear',C=1.0)
svm_lda.fit(X_train_lda, y_train)

#KPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.0769)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)
log_kpca=LogisticRegression(multi_class='ovr')
log_kpca.fit(X_train_kpca,y_train)
svm_kpca=SVC(kernel='linear',C=1.0)
svm_kpca.fit(X_train_kpca, y_train)

model = {'log': [log, X_train, X_test],
         'svm':[svm, X_train, X_test], 
         'log_pca':[log_pca, X_train_pca, X_test_pca], 
         'svm_pca':[svm_pca, X_train_pca, X_test_pca],
         'log_lda':[log_lda, X_train_lda, X_test_lda],
         'svm_lda':[svm_lda,X_train_lda, X_test_lda], 
         'log_kpca': [log_kpca, X_train_kpca, X_test_kpca],
         'svm_kpca': [svm_kpca, X_train_kpca, X_test_kpca]}

for i in model:
    print('For the model ', i )
    scores = accuracy_results(model[i][0], model[i][1], y_train, model[i][2], y_test)
    print('Accuracy Scores')
    print('CV: %.4f, train: %.4f, test: %.4f' % (scores[0], scores[1],scores[2]))
    print()
    
plt.figure()
plot_decision_regions(X_test_pca, y_test, log_pca,test_idx=None, resolution=0.02)
plt.legend(loc='lower left') 
plt.figure()
plot_decision_regions(X_test_pca, y_test, svm_pca,test_idx=None, resolution=0.02)
plt.legend(loc='lower left')
plt.figure()
plot_decision_regions(X_test_lda, y_test, log_lda,test_idx=None, resolution=0.02)
plt.legend(loc='lower left')
plt.figure()
plot_decision_regions(X_test_lda, y_test, svm_lda,test_idx=None, resolution=0.02)
plt.legend(loc='lower left')
plt.figure()
plot_decision_regions(X_test_kpca, y_test, log_kpca,test_idx=None, resolution=0.02)
plt.legend(loc='lower left')
plt.figure()
plot_decision_regions(X_test_kpca, y_test, svm_kpca,test_idx=None, resolution=0.02)
plt.legend(loc='lower left')

score_log=[]
score_svm=[]
gamma_range=np.arange(0.05,1,0.05)
for g in gamma_range:
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)
    X_train_kpca = kpca.fit_transform(X_train)
    X_test_kpca = kpca.transform(X_test)
    log_kpca=LogisticRegression(multi_class='ovr')
    log_kpca.fit(X_train_kpca,y_train)
    svm_kpca=SVC(kernel='linear',C=1.0)
    svm_kpca.fit(X_train_kpca, y_train)
    score_log.append(accuracy_results(log_kpca, X_train_kpca, y_train, X_test_kpca, y_test)[2])
    score_svm.append(accuracy_results(svm_kpca, X_train_kpca, y_train, X_test_kpca, y_test)[2])

best_acc = max(score_log)
index = [i for i,x in enumerate(score_log) if x==best_acc]
print(gamma_range[index])
best_acc = max(score_svm)
index = [i for i,x in enumerate(score_svm) if x==best_acc]
print(gamma_range[index])

print("My name is Bingjie Han")
print("My NetID is: bingjie5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
