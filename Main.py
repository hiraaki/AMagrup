import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import math
import GroupTest as gt


data_frame = pd.read_csv("Maurício.csv")
x = data_frame.iloc[:, :2].values
y = data_frame.iloc[:, 2:].values
# gt.bestfitKmeans(data_frame)
coe = []
ent = []
sep = []
csi = []
for i in range(10):
    kmeans = KMeans(n_clusters=20, init='random', n_init=20, max_iter=60).fit(x) #para coesão
    labels = kmeans.labels_
    grupos = gt.getgroups(20, labels, x)
    coe.append(gt.coesao(grupos))

    kmeans = KMeans(n_clusters=18, init='random', n_init=4, max_iter=80).fit(x) #para separabilidade
    labels = kmeans.labels_
    grupos = gt.getgroups(18, labels, x)
    sep.append(gt.separabilidade(grupos))

    kmeans = KMeans(n_clusters=18, init='random', n_init=4, max_iter=80).fit(x)  # para separabilidade
    labels = kmeans.labels_
    grupos = gt.getgroups(18, labels, x)
    csi.append(gt.coe_Sirueta(grupos))

print("Resultado Kmeans")
print("coesão: ", np.mean(coe).__str__())
print("entropia: 0")
print("eparabilidade: ", (np.mean(sep)).__str__())
print("coeficiente: ", (np.mean(csi)).__str__())

coe.clear()
ent.clear()
sep.clear()
csi.clear()
clas = []
throwed = []
while len(ent) < 10:
    throw = []
    cla = []
    dbscan = DBSCAN(eps=4, min_samples=2).fit(x)
    labels = dbscan.fit_predict(x)
    while 1 > max(labels):
        dbscan = DBSCAN(eps=6, min_samples=4).fit(x)
        labels = dbscan.fit_predict(x)
        print(max(labels))
    grupos = gt.getgroups(max(labels) + 1, labels, x)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data_frame.index.values
    cluster_map['classe'] = data_frame.classe.values
    cluster_map['cluster'] = labels
    throw.append(cluster_map.query('cluster == -1').count()[0])
    cla.append(max(labels)+1)
    coe.append(gt.coesao(grupos))

    dbscan = DBSCAN(eps=6, min_samples=4).fit(x)
    labels = dbscan.fit_predict(x)
    while 1 > max(labels):
        dbscan = DBSCAN(eps=6, min_samples=4).fit(x)
        labels = dbscan.fit_predict(x)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data_frame.index.values
    cluster_map['classe'] = data_frame.classe.values
    cluster_map['cluster'] = labels
    clas.append(max(labels) + 1)
    throw.append(cluster_map.query('cluster == -1').count()[0])
    ent.append(gt.entropia(cluster_map))

    dbscan = DBSCAN(eps=196, min_samples=2).fit(x)
    labels = dbscan.fit_predict(x)
    while 1 > max(labels):
        dbscan = DBSCAN(eps=6, min_samples=4).fit(x)
        labels = dbscan.fit_predict(x)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data_frame.index.values
    cluster_map['classe'] = data_frame.classe.values
    cluster_map['cluster'] = labels
    cla.append(max(labels) + 1)
    throw.append(cluster_map.query('cluster == -1').count()[0])
    sep.append(gt.separabilidade(grupos))

    dbscan = DBSCAN(eps=196, min_samples=2).fit(x)
    labels = dbscan.fit_predict(x)
    while 1 > max(labels):
        dbscan = DBSCAN(eps=6, min_samples=4).fit(x)
        labels = dbscan.fit_predict(x)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data_frame.index.values
    cluster_map['classe'] = data_frame.classe.values
    cluster_map['cluster'] = labels
    cla.append(max(labels) + 1)
    throw.append(cluster_map.query('cluster == -1').count()[0])
    csi.append(gt.coe_Sirueta(grupos))
    throwed.append(throw)
    clas.append(cla)

print("Resultado DBScan")
print("Ruido", throwed.__str__())
print("Qtd de Clas", clas.__str__())
print("coesão: ", np.mean(coe).__str__())
print("entropia: 0")
print("eparabilidade: ", np.mean(sep).__str__())
print("coeficiente: ", np.mean(csi).__str__())

coe.clear()
ent.clear()
sep.clear()
csi.clear()
clas.clear()
for i in range(10):
    cla = []
    agnes = AgglomerativeClustering(linkage="complete").fit(x)
    labels = agnes.labels_
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data_frame.index.values
    cluster_map['classe'] = data_frame.classe.values
    cluster_map['cluster'] = labels
    grupos = gt.getgroups(max(labels)+1, labels, x)
    cla.append(max(labels)+1)
    coe.append(gt.coesao(grupos))
    ent.append(gt.entropia(cluster_map))
    sep.append(gt.separabilidade(grupos))
    csi.append(gt.coe_Sirueta(grupos))
print("Resultado Agnes")
print("Qtd de Clas", clas.__str__())
print("coesão: ", np.mean(coe).__str__())
print("entropia: 0")
print("eparabilidade: ", np.mean(sep).__str__())
print("coeficiente: ", np.mean(csi).__str__())
