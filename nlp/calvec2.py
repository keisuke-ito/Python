import csv
import numpy as np
import xlrd 
import openpyxl 
from gensim.models import word2vec 

#保存先のファイルを作成
csvFile = open("adj_sntex1.csv", 'wt', newline = '', encoding = 'utf-8')
writer = csv.writer(csvFile)

model = word2vec.Word2Vec.load('enwiki10.model') 

#距離行列を求める単語群のデータを開く
wb = xlrd.open_workbook('Chemicals_adjex.xlsm') #refer
sheets = wb.sheets()
sheet = wb.sheet_by_index(0)

row = [[None for p in range(sheet.ncols)] for q in range(sheet.nrows)]
col = [[None for p in range(sheet.ncols)] for q in range(sheet.nrows)]

for y in range(sheet.nrows):
    for x in range(sheet.ncols):
        row[y][x]=sheet.cell(y,x).value
col=row

#"1－単語間のコサイン距離"を計算
vec=[0]*sheet.nrows
for k in range(sheet.nrows):
    for i in range(sheet.nrows):
        try:
            if k==i :
                vec[i] = 0
            else:
                vec[i] = 1-model.similarity(row[k][0], col[i][0])
        except KeyError:
            pass
    print(vec)
    writer.writerow(vec)

    
