import gensim.models
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('data.txt')   #コーパスの選択

model = word2vec.Word2Vec(sentences, size=400, min_count=5, window=10)
#単語ベクトルの次元数(size),文章とみなす最少単語数(min_count),周辺単語数(window)
model.save("./wiki.model")