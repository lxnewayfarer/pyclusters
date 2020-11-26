from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib

# matplotlib lines width
matplotlib.rcParams['lines.linewidth'] = 3

# read CSV file Сервис;Сценарий;Время отправки;Номер;Статус;Время получения;Отправитель;ID сообщения;Текст;Кол-во частей;Скорость доставки
raw = pd.read_csv('data.csv', delimiter=';')

# take only delivered messages
delivered = raw[raw['Статус'] == 'Доставлено']

# plot data_labels idx, sender
data_labels = list(str(x) + '. ' + delivered['Отправитель'].values[x]
                   for x in range(len(delivered['Отправитель'].values)))

# print data labels
for idx, x in enumerate(delivered[['Отправитель', 'Текст']].values):
    print(idx, x)

# corpus = texts of all delivered messages 
corpus = list(x[8] for idx, x in delivered.iterrows())

# making dataset
dataset = pd.DataFrame(corpus, columns=['des'])

# making TF-IDF model
# token_pattern - only words
# min_df - if token's document frequency < 0,03 this token is not counted
tfidf = TfidfVectorizer(analyzer='word',
                        token_pattern=r'[a-zA-Z]+',
                        min_df=0.03
                        )
# vectors dataframe
tf_idf_matrix = pd.DataFrame(
    tfidf.fit_transform(dataset['des']).toarray(),
    columns=tfidf.get_feature_names()
)

# print dataframe head
print(tf_idf_matrix.head())

# doing hierarchical clustering
linked = linkage(tf_idf_matrix.values, 'centroid')

print('Distances between merged clusters on each iteration:\n', linked[:, 2])

# show dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=data_labels, leaf_font_size=10,
           distance_sort='descending', show_leaf_counts=True)
           
plt.show()
