import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import sys
def clusterIndicesNumpy(clustNum, labels_array):
    return np.where(labels_array == clustNum)[0]


def getgrouplabel(indexies, lista):
    lista_ = []
    for index in indexies:
        lista_.append((lista[index]))
    return lista_


def getgroups(ngroups, labels, lista):
    groups = []
    for i in range(ngroups):
        print(i)
        aux = clusterIndicesNumpy(i, labels)
        aux2 = getgrouplabel(aux, lista)
        groups.append(aux2)
    return groups

def getgroups(ngroups, labels, lista):
    groups = []
    for i in range(ngroups):
        aux = clusterIndicesNumpy(i, labels)
        aux2 = getgrouplabel(aux, lista)
        groups.append(aux2)
    return groups


def euclidiana(origem, fim):
    dif = []
    for i in range(len(origem)):
        aux = pow((origem[i] - fim[i]), 2)
        dif.append(aux)
    return math.sqrt(sum(dif))


def umPontoeUmGrupo(ponto, grupo):
    distancias = []
    for i in range(len(grupo)):
        distancias.append(euclidiana(ponto, grupo[i]))
    return distancias


def umPontoeUmGrupoQuadrado(ponto, grupo):
    distancias = umPontoeUmGrupo(ponto, grupo)
    distanciaQ = []
    for d in range(len(distancias)):
        distanciaQ.append(distancias[d]**2)
    return np.mean(distanciaQ)


def coesao(clusters):
    somatosrios = []
    # print(len(clusters))
    for cluster in clusters:
        # print(len(cluster))
        soma = []
        for i in range(len(cluster)):
            for j in range(len(cluster)):
                if i != j:
                    soma.append(pow(euclidiana(cluster[i], cluster[j]), 2))
        somatosrios.append(np.mean(soma))
    return sum(somatosrios)/len(clusters)


def entropia(cluster_map):
    entropias = []
    for i in range(cluster_map.max()[2]+1):
        entropia = []
        cluster = cluster_map.query(('cluster =='+i.__str__()))
        umC = cluster[cluster["classe"] == 1].count()
        doisC = cluster[cluster["classe"] == 2].count()
        # print(umC)
        # print(doisC)
        if (umC[0] == 0) & (doisC[0] > 0):
            # print("só tem dois")
            entropias.append(0)
        if (umC[0] > 0) & (doisC[0] == 0):
            # print("só tem um")
            entropias.append(0)
        if (umC[0] > 0) & (doisC[0] > 0):
            # print("tem dos dois")
            entropia.append(((umC[0] / (umC[0] + doisC[0])) * (math.log((umC[0]/(umC[0]+doisC[0])), 2))))
            entropia.append(((doisC[0] / (umC[0] + doisC[0])) * (math.log((doisC[0]/(umC[0]+doisC[0])), 2))))
            entropias.append(np.mean(entropia)*-1)

        # print(entropias[len(entropias)-1])

    # print(np.mean(entropias))
    return np.mean(entropias)


def separabilidade(clusters):
    # print(len(clusters))
    dgrupos = []
    for i in range(len(clusters)): #para cada cluster
        clusterS = clusters[i]
        distancias = []
        for j in range(len(clusterS)): #para cada menbro de grupoS
            for k in range(len(clusters)): #adiciona a soma das distancias entre o membro de grupoS e os membros dos outros grupos
                if i != k:
                    distancias.append(umPontoeUmGrupoQuadrado(clusterS[j], clusters[k])) #soma das distâncias ao quadrado de cada grupo
        dgrupos.append(np.sum(distancias)) #média do somatório das distnacias ao quadrada entre um cluster e os restantes
    return np.mean(dgrupos) #média geral das distâncias ao quadrado


def coe_Sirueta(clusters):
    # print(len(clusters))
    csg = []
    # print(len(clusters))
    for i in range(len(clusters)):
        cs = []
        clusterS = clusters[i]
        for j in range(len(clusterS)):
            somaInterno = np.mean(umPontoeUmGrupo(clusterS[j], clusterS))
            distancias = []
            for k in range(len(clusters)):
                if i != k:
                    distancias.append(np.mean(umPontoeUmGrupo(clusterS[j], clusters[k])))
            distancias.sort()
            cs.append((distancias[0]-somaInterno)/max(distancias[0], somaInterno))
        csg.append(np.mean(cs))
    return np.mean(csg)


def bestfitKmeans(data_frame):
    x = data_frame.iloc[:, :2].values
    y = data_frame.iloc[:, 2:].values
    executiontable = pd.DataFrame(columns=['Cluster', 'Sementes', 'Iterações', 'Coesão', 'Entropia', 'Separabilidade', 'Coeficiente de Sirueta'])
    print("--------------executando kmeans-------------------------")
    bestcoe = [sys.maxsize, 0, 0, 0]
    bestsep = [-1, 0, 0, 0]
    bestent = [sys.maxsize, 0, 0, 0]
    bestcsi = [sys.maxsize, 0, 0, 0]

    for i in range(2, 20): #numero de clusters
        print(i)
        for j in range(4, 20, 4): #numero de tentativas em sementes diferentes
            for k in range(20, 200, 20): #numedo de iterações
                kmeans = KMeans(n_clusters=i, init='random', n_init=j, max_iter=k).fit(x)
                cluster_map = pd.DataFrame()
                cluster_map['data_index'] = data_frame.index.values
                cluster_map['classe'] = data_frame.classe.values
                cluster_map['cluster'] = kmeans.labels_
                grupos = getgroups(i, kmeans.predict(x), x)
                coe = coesao(grupos)
                ent = entropia(cluster_map)
                sep = separabilidade(grupos)
                csi = coe_Sirueta(grupos)
                if coe <= bestcoe[0]:
                    bestcoe[0] = coe
                    bestcoe[1] = i
                    bestcoe[2] = j
                    bestcoe[3] = k
                if ent <= bestent[0]:
                    bestent[0] = ent
                    bestent[1] = i
                    bestent[2] = j
                    bestent[3] = k
                if sep >= bestsep[0]:
                    bestsep[0] = sep
                    bestsep[1] = i
                    bestsep[2] = j
                    bestsep[3] = k
                if (csi >= 0) & (csi <= bestcsi[0]):
                    bestcsi[0] = csi
                    bestcsi[1] = i
                    bestcsi[2] = j
                    bestcsi[3] = k

                executiontable=pd.DataFrame(np.array([[i,j,k,coe,ent,sep,csi]]), columns=('Cluster', 'Sementes', 'Iterações', 'Coesão', 'Entropia', 'Separabilidade', 'Coeficiente de Sirueta')).append(executiontable,ignore_index=True)

    print(bestcoe)
    print(bestent)
    print(bestsep)
    print(bestcsi)
    executiontable.to_csv("Execução kmeans")


def bestfitDBScan(data_frame):
    x = data_frame.iloc[:, :2].values
    y = data_frame.iloc[:, 2:].values
    executiontable = pd.DataFrame(columns=['Raio', 'Minimo P', 'Coesão', 'Entropia', 'Separabilidade', 'Coeficiente de Sirueta'])
    print("--------------executando DBSCAN-------------------------")
    bestcoe = [math.inf, 0, 0]
    bestsep = [-math.inf, 0, 0]
    bestent = [math.inf, 0, 0]
    bestcsi = [math.inf, 0, 0]
    # print(data_frame)
    for i in range(2, 200): #Tamanho do raio
        for j in range(2, 200): #numero minimo de pontos
            dbscan = DBSCAN(eps=i, min_samples=j, metric='euclidean')
            labels = dbscan.fit_predict(x)
            cluster_map = pd.DataFrame()
            cluster_map['data_index'] = data_frame.index.values
            cluster_map['classe'] = data_frame.classe.values
            cluster_map['cluster'] = labels
            if (cluster_map.query('cluster != -1').count()[0] > 1) & (max(labels)>1):
                grupos = getgroups(max(labels)+1, labels, x)
                # print(cluster_map.query('cluster != -1'))
                # print(grupos)
                coe = coesao(grupos)
                if coe <= bestcoe[0]:
                    bestcoe[0] = coe
                    bestcoe[1] = i
                    bestcoe[2] = j


                ent = entropia(cluster_map)
                if ent < bestent[0]:
                    bestent[0] = ent
                    bestent[1] = i
                    bestent[2] = j


                sep = separabilidade(grupos)
                if sep > bestsep[0]:
                    bestsep[0] = sep
                    bestsep[1] = i
                    bestsep[2] = j


                csi = coe_Sirueta(grupos)
                if (csi >= 0) & (csi < bestcsi[0]):
                    bestcsi[0] = csi
                    bestcsi[1] = i
                    bestcsi[2] = j
                executiontable = pd.DataFrame(np.array([[i,j,coe,ent,sep,csi]]), columns=('Raio', 'Minimo P', 'Coesão', 'Entropia', 'Separabilidade', 'Coeficiente de Sirueta')).append(executiontable,ignore_index=True)
                print(bestcoe)
                print(bestent)
                print(bestsep)
                print(bestcsi)
    executiontable.to_csv("Execução DBSCAN")




