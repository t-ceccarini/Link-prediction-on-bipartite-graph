from networkit import *
from collections import namedtuple
from tempfile import TemporaryFile

import matplotlib.pyplot as plt 
import numpy as np
import networkx as nx

def getSubGraph(G, size, numUtenti):
	sumDegree = 0
	G_sample = Graph(G, weighted=True)
	removed_edges = list()
	for i in range(0, numUtenti):
		sumDegree += G.degree(i)

	for i in range(0, numUtenti):
		sampleSize = int(np.trunc((G.degree(i) / sumDegree)*size))
		for j in range(0, sampleSize):
			flag = False
			for n in G_sample.neighbors(i):
				if G_sample.weight(i,n) == 1:
					flag = True
			if flag == False:
				break
			u = G_sample.randomNeighbor(i)
			while(G_sample.weight(i,u) < 1):
				u = G_sample.randomNeighbor(i)
			G_sample.removeEdge(i,u)
			removed_edges.append((u,i))
	return G_sample,removed_edges


def readGraph(path, n):
	file = open(path, "r")
	G = Graph(n, weighted = True)
	for line in file:
		edge = line.split()
		G.addEdge(int(edge[0]),int(edge[1]),w = int(edge[2]))
	file.close()
	return G

def fromNumpyMatrix(M,G):
	G_k = Graph(n = np.size(M,0),weighted = True,directed = False)
	for i in range(0,np.size(M,0)):
		for j in range(0,np.size(M,1)):
			#if G.hasEdge(i,j):
			G_k.addEdge(i,j,M[i,j])
	return G_k

def commonNeighborsFilm(G,y,x):
	x_neighbors = G.neighbors(x)
	y_neighbors = G.neighbors(y)
	neighborhood_of_y_neighbors = list()
	for y_neighbor in y_neighbors:
		neighborhood_of_y_neighbors.extend(G.neighbors(y_neighbor)) 
	CN = list(set(neighborhood_of_y_neighbors).intersection(x_neighbors))

	s = 0.0

	for node in CN:
		node_neighbors = G.neighbors(node)
		neighborhood_of_y_and_of_node = list(set(node_neighbors).intersection(y_neighbors))
		for user in neighborhood_of_y_and_of_node:
			s += G.weight(user,node)
		s += G.weight(x,node)	
	CN_weighted_index = s
	return CN_weighted_index


def similarityMatrix(G, numUtenti):
	S = np.zeros(shape=(numUtenti,numUtenti))
	for i in range(0,numUtenti):
		i_neighbors = G.neighbors(i)
		for j in range(0,numUtenti):
			s = 0.0
			j_neighbors = G.neighbors(j)
			i_j_cn = list(set(i_neighbors).intersection(j_neighbors))
			for k in i_j_cn:
				s += G.weight(i,k)*G.weight(j,k)
			s = s / (len(list(set(i_neighbors).union(j_neighbors))))
			S[i,j] = s
	return S		

def score_cn(G,x,u,S):
	x_neighbors = G.neighbors(x)
	norm = 0.0
	for v in x_neighbors:
		norm += abs(S[u,v])
	s = 0.0
	for v in x_neighbors:
		s += G.weight(v,x)*S[u,v]
	cn_index = s / norm
	cn_index = s
	return cn_index


def computeKatzIndex(M,beta,l):
	score = np.matrix(beta*M)
	M_t = M
	for i in range(2,l+1):
		M_t = M_t*M
		score += score*pow(beta,i)*M_t
	return score

def low_rank_approx(SVD=None, A=None, r=1):
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar


G_f = readGraph('valutazioni-finali2.txt', 135)
G, removed_edges = getSubGraph(G_f,200, 31)
n = len(removed_edges)

M = algebraic.adjacencyMatrix(G, matrixType='dense')

u, s, v = np.linalg.svd(np.matrix(M), full_matrices=False)

u[abs(u) < 0.001] = 0
s[abs(s) < 0.001] = 0
v[abs(v) < 0.001] = 0

M_k = low_rank_approx((u,s,v), r=12)
M_k[abs(M_k) < 0.001] = 0
G_k = fromNumpyMatrix(M_k, G)
E = G.edges()

katz= linkprediction.KatzIndex(G,5,0.05)
katz_approx= linkprediction.KatzIndex(G_k,5,0.05)
random = linkprediction.AdjustedRandIndex(G)

X = G.nodes()[0:31]
Y = G.nodes()[31:135]

C = [(i,j) for i in Y for j in X]

training_set = list(set(C)-set(E))

Sim = similarityMatrix(G, 31) #calcolo matrice di similarità 

#pred_cn = [(c,commonNeighborsFilm(G,c[0],c[1])) for c in training_set ]
pred_cn = [(c,score_cn(G,c[0],c[1],Sim)) for c in training_set ]
pred_katz = [(c,katz.run(c[0],c[1])) for c in training_set ]
pred_katz_approx = [(c,katz_approx.run(c[0],c[1])) for c in training_set ]
pred_random = [(c,random.run(c[0],c[1])) for c in training_set ]

lt = linkprediction.LinkThresholder()

rank_cn = lt.byCount(pred_cn,n)
rank_katz = lt.byCount(pred_katz, n)
rank_katz_approx = lt.byCount(pred_katz_approx, n)
rank_random = lt.byCount(pred_random, n)

print("percentuali corrette")

p_cn = len(set(rank_cn).intersection(set(removed_edges)))/n
p_katz = len(set(rank_katz).intersection(set(removed_edges)))/n
p_katz_k = len(set(rank_katz_approx).intersection(set(removed_edges)))/n 
p_random = len(set(rank_random).intersection(set(removed_edges)))/n 

print("CN "+str(p_cn*100)+"%")
print("Katz "+str(p_katz*100)+"%")
print("Katz-approx "+str(p_katz_k*100)+"%")
print("Random "+str(p_random*100)+"%")
