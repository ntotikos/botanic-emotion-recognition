""" Here, different plots for the data are created."""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

from src.data.dataset_builder import EkmanDataset
from src.utils.constants import DATASETS_DIR
from src.utils.reproducibility import set_seed

set_seed(42)

# Get the TS dataset.
path_to_pickle = DATASETS_DIR / "sdm_2023-01_all_valid_files_version_1.pkl"
dataset = EkmanDataset(path_to_pickle)
dataset.get_data_and_labels()

print(len(dataset.dataset))

data_shape = dataset.dataset.tensors[0].shape
label_shape = dataset.dataset.tensors[1].shape

print(data_shape)
print(label_shape)

data_tensor = dataset.dataset.tensors[0]
label_tensor = dataset.dataset.tensors[1]

X = data_tensor.numpy()
y = label_tensor.numpy()

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
#
# plt.figure(figsize=(10, 8))
# for label in np.unique(y):
#     plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=f"Class {label}")
# plt.legend()
# plt.title('PCA Visualization')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.savefig('pca_visualization_untouched.png', dpi=300, bbox_inches='tight')
# plt.show()

tsne = TSNE(n_components=2, random_state=42, verbose=1, n_iter=1000)
X_tsne = tsne.fit_transform(X)

np.save('tsne_values_1000.npy', X_tsne)

plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=label)
plt.legend()
plt.title('t-SNE Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('tsne_visualization_untouched.png', dpi=300, bbox_inches='tight')
plt.show()

