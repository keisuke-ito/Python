import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np
import xlrd
import openpyxl
from sklearn.cluster import KMeans
import pprint
import scipy.spatial.distance as distance
from random import random
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist 
from scipy.cluster.hierarchy import linkage, dendrogram , fcluster, set_link_color_palette
from matplotlib import pyplot as plt


#距離行列のデータセットを開く
wb = xlrd.open_workbook('clu_15.xlsx')
sheets = wb.sheets()
sheet = wb.sheet_by_index(0)

#行列にエクセルより得た値を格納
dMatrix = [[None for p in range(sheet.ncols)] for q in range(sheet.nrows)]

for y in range(sheet.nrows):
    for x in range(sheet.ncols):
        dMatrix[y][x]=sheet.cell(y,x).value

#label_name
sheet_name = wb.sheet_by_index(1)
name= [[None for p in range(sheet_name.ncols)] for q in range(sheet_name.nrows)]
for y in range(sheet_name.nrows):
    for x in range(sheet_name.ncols):
        name[y]=sheet_name.cell(y,x).value

        
result=linkage(dMatrix,method='ward')
r=dendrogram(result, p=750,truncate_mode='lastp',labels=name)
group = fcluster(result, 3.0, criterion='distance') # ユークリッド平方距離で分けたい場合
#group = fcluster(result,13 , criterion='maxclust') # クラスタ数で分けたい場合

print(group)
#plt.xlabel("xlabel", fontsize=25)
#print(r["leaves"]);
#print(r["ivl"]);
#plt.show()
