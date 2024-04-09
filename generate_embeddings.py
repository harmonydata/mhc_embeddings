import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle as pkl

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

with open("mhc_questions.txt", "r", encoding="utf-8") as f:
    texts = np.asarray(list(f))


embeddings = model.encode(texts)

embeddings_small = np.float16(embeddings)
with open("mhc_embeddings.npy", "wb") as f:
    pkl.dump(embeddings_small, f)
