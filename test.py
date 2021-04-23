import utils.dataload as dl
from sklearn.model_selection  import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, tree
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import datetime
import seaborn as sn

#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
def clf_byss():
    time1 = datetime.datetime.now()
    clf_bysp = MultinomialNB()
    clf_bys = clf_bysp.fit(X_train,y_train)
    predict_labels = clf_bys.predict(X_test)
    time2 = datetime.datetime.now()
    return predict_labels, y_test, time2-time1

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
def clf_svms():
    time1 = datetime.datetime.now()
    clf_svmp = svm.SVC(kernel='linear', C=0.01)
    clf_svm = clf_svmp.fit(X_train,y_train)
    predict_labels = clf_svm.predict(X_test)
    time2 = datetime.datetime.now()
    return predict_labels, y_test, time2-time1

#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontree#sklearn.tree.DecisionTreeClassifier
def clf_dtrees():
    time1 = datetime.datetime.now()
    clf_dtreep = tree.DecisionTreeClassifier(max_depth=10)
    clf_dtree = clf_dtreep.fit(X_train,y_train)
    predict_labels = clf_dtree.predict(X_test)
    time2 = datetime.datetime.now()
    return predict_labels, y_test, time2-time1

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?highlight=kmeans#sklearn.cluster.KMeans
def clf_kms():
    X,Y = dl.DataLoader(path, labels, relabel='true').data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    time1 = datetime.datetime.now()
    clf_kmp = KMeans(n_clusters=10,max_iter=300)
    clf_km = clf_kmp.fit(X_train, Y)
    predict_labels = clf_km.predict(X_test)
    time2 = datetime.datetime.now()
    return  predict_labels, y_test, time2-time1


if __name__ == "__main__":
    # 类别标签：人，沙滩，建筑，卡车，恐龙，大象，花朵，马，山峰，食品
    labels = ['people', 'beaches', 'buildings', 'trucks', 'dinosaurs', 'elephants', 'flowers', 'horses', 'mountains', 'food']
    # 图像数据处理
    path = "Sort_1000pics"
    X, Y = dl.DataLoader(path, labels).data()
    # 将数据分为训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    result=[]
    time = datetime.datetime.now()
    with open('./result/OutputsH.txt', 'a') as f:
        f.write('{}\n'.format(time))
    titles_options = [("Confusionmatrix_Bayes", clf_byss()),
                      ('Confusionmatrix_SVM', clf_svms()),
                      ('Confusionmatrix_Decision-Tree', clf_dtrees()),
                      ('Confusionmatrix_KMeans', clf_kms())]
    for title, result in titles_options:
        predict_labels, y_test, cost = result[0], result[1], result[2]
        con_mat = confusion_matrix(y_test, predict_labels)  #混淆矩阵
        con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        reports = classification_report(y_test, predict_labels, target_names=labels, zero_division=1) #测试报告
        accuracy_scores = accuracy_score(y_test, predict_labels)
        #https: // seaborn.pydata.org / generated / seaborn.heatmap.html
        plt.figure(figsize=(12, 12), dpi=120)#可视化
        # plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        # plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        plt.text(11,-1, 'Acc: {:.2f}'.format(accuracy_scores), fontdict={'fontsize': 15,'fontweight' : 1000})
        ax = sn.heatmap(con_mat_norm, annot=True, xticklabels=labels,yticklabels=labels,cmap=plt.cm.Blues )
        ax.xaxis.set_ticks_position('top')
        ax.set_title(title, fontdict={'fontsize': 20,'fontweight' : 1000})  # 标题
        ax.set_xlabel('Predict Label', fontdict={'fontsize': 15,'fontweight' : 1000})  # x轴
        ax.set_ylabel('True Label', fontdict={'fontsize': 15,'fontweight' : 1000})  # y轴
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix
        # ConfusionMatrixDisplay(con_mat_norm, display_labels=labels).plot(cmap=plt.cm.Blues,xticks_rotation=35).ax_.set_title(title)
        #结果存储
        with open('./result/OutputsH.txt','a') as f:
            f.write('{} Report:\n{}\nCost:\n{}\n\n\n'.format(title[16::], reports, cost))
        # print('{} Report:\n{}\n\tCost:\n\t{}\n\n'.format(title[16::], reports, cost))
        plt.savefig('./result/{}.jpg'.format(title[16::],time))
    f.close()