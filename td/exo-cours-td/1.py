# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
corpus = [
    "le chat mange la souris",
    "le chien chasse le chat",
    "la souris devore le fromage"
]
contextes = {}

tfidf = TfidfVectorizer()
corpus_vector = tfidf.fit_transform(corpus)

index = [f"phrase {i+1}" for i in range(len(corpus))]
pd.DataFrame(corpus_vector.toarray(), index=index, columns=tfidf.get_feature_names_out())


#%%
from sklearn.metrics.pairwise import cosine_similarity as cosine
cosine(tfidf.transform(["Le chat mange la souris"]).toarray(), 
       tfidf.transform(["Le chien chasse le chat"]).toarray())
# %%