""" Here, different plots for the data are created."""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

from src.data.dataset_builder import EkmanDataset
from src.utils.constants import DATASETS_DIR
from src.utils.reproducibility import set_seed
from src.utils.constants import INT_TO_EKMAN_DICT

set_seed(42)

# Get the TS dataset.
path_to_pickle = DATASETS_DIR / "sdm_2023-01_all_valid_files_version_1.pkl"
dataset = EkmanDataset(path_to_pickle)
dataset.get_data_and_labels_without_neutral()
dataset.normalize_samples(normalization="min-max-scaling")

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
# plt.savefig('pca_visualization_untouched_without_neutral_min_max.png', dpi=300, bbox_inches='tight')
# plt.show()

# pca = PCA(n_components=3)
# X_pca = pca.fit_transform(X)
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# for label in np.unique(y):
#     indices = np.where(y == label)
#     ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2], label=label)
#
# ax.set_title("3D PCA of the Data")
# ax.set_xlabel("Principal Component 1")
# ax.set_ylabel("Principal Component 2")
# ax.set_zlabel("Principal Component 3")
# ax.legend()
# plt.show()

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

num_samples_per_class = 150

for label in np.unique(y):
    indices = np.where(y == label)[0]
    selected_indices = np.random.choice(indices, size=min(num_samples_per_class, len(indices)), replace=False)
    ax.scatter(X_pca[selected_indices, 0], X_pca[selected_indices, 1], X_pca[selected_indices, 2],
               label=INT_TO_EKMAN_DICT[label])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.savefig('pca_visualization_untouched_without_neutral_min_max_100samples.png', dpi=300, bbox_inches='tight')
plt.show()









# tsne = TSNE(n_components=2, random_state=42, verbose=1, n_iter=2000)
# X_tsne = tsne.fit_transform(X)
#
# np.save('tsne_values_2000.npy', X_tsne)
#
# plt.figure(figsize=(8, 6))
# for label in np.unique(y):
#     plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=label)
# plt.legend()
# plt.title('t-SNE Visualization')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.savefig('tsne_visualization_untouched_without_neutral_minmax.png', dpi=300, bbox_inches='tight')
# plt.show()



