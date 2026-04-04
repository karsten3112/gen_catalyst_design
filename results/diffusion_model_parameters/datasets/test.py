
from sklearn.manifold import TSNE
import numpy as np

X = np.random.RandomState(42).randn(200, 50).astype(np.float64)
Y = TSNE(
    n_components=2,
    random_state=42,
    init="pca",
    learning_rate=200.0,
    perplexity=30,
).fit_transform(X)

print(Y.shape)