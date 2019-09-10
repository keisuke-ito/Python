# 5-Layer Autoencoder

このフォルダは５層の深層オートエンコーダを作成した際のソースコードの保存用

## Purpose
マススペクトルデータの次元圧縮(特徴抽出)

## データセット
データはマススペクトルデータ(MS data)を採用している．  
今回使用したMSデータは  
米国 National Institute of Standards and Technology (NIST)がインターネット上で公開する
NIST Chemistry WebBookのデータを使用．  
NISTが公開するこのデータベースにおいて、サンプルのマススペクトルは「EI 法(70eV)」の条件のもとに測定されている.  
NIST Chemistry WebBookでは15万種類以上の化学物質について、マススペクトルに加えて以下のデータが提供されている.  

1. Formula  
2. Molecular weight  
3. IUPAC Standard InChI  
4. IUPAC Standard InChIKey  
5. CAS Registry Numbe  
6. Chemical structure (2D and 3D)  
7. Other names  
8. Gas phase thermochemistry data  
9. Phase change data  
10. Reaction thermochemistry data  
11. Henry's Law data  
12. Gas phase ion energetics data  
13. Ion clustering data  
14. Mass spectrum (electron ionization)  
15. Constants of diatomic molecules  
16. Fluid Properties  
17. UV/Visible spectrum  
18. Vibrational and/or electronic energy levels (19) Gas Chromatography  