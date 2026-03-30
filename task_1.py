import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode
import numpy as np

iris = load_iris() 
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("Staty zbioru")
print(df.describe())
df['Gatunek'] = pd.Categorical.from_codes(y, iris.target_names)

# część do wykresów macierzowych i rozkładu normalnego

pairplot_fig = sns.pairplot(
    df.drop(columns=['target']),
    hue='Gatunek', 
    diag_kind='kde',
    palette='pastel',
    markers=["o", "s", "D"]
)
pairplot_fig.figure.suptitle("Zestawienie wykresów", y=1.02, fontsize=16)
pairplot_fig.savefig("iris_pairplot_eda.png", bbox_inches='tight')
plt.close()

kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_3 = kmeans_3.fit_predict(X)
labels_3 = np.zeros_like(clusters_3)
for i in range(3):
    mask = (clusters_3 == i)
    if np.any(mask):
        labels_3[mask] = mode(y[mask], keepdims=True)[0][0]

acc_3 = accuracy_score(y, labels_3)
prec_3 = precision_score(y, labels_3, average='macro')
rec_3 = recall_score(y, labels_3, average='macro')
f1_3 = f1_score(y, labels_3, average='macro')

print("\nWyniki po parametrze k=3")
print(f"Accuracy:  {acc_3:.4f}")
print(f"Precision: {prec_3:.4f}")
print(f"Recall:    {rec_3:.4f}")
print(f"F1-score:  {f1_3:.4f}")

# część do wykresu t-sne

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(X)
cluster_legend_names = [f"Grupa: {iris.target_names[label]}" for label in labels_3]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=tsne_results[:, 0], 
    y=tsne_results[:, 1],
    hue=cluster_legend_names, 
    palette="viridis",
    s=100,
    alpha=0.8
)
plt.title("Wizualizacja T-SNE po grupowaniu K-Means (Zbiór Iris)")
plt.xlabel("T-SNE Cecha 1")
plt.ylabel("T-SNE Cecha 2")
plt.legend(title="Grupy w wizualizacji")
plt.grid(True)
plt.savefig("tsne_kmeans_iris.png")
plt.close() 

k_values = range(1, 11)
accuracies = []
precisions = []
recalls = []
f1_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    labels = np.zeros_like(clusters)
    for i in range(k):
        mask = (clusters == i)
        if np.any(mask):
            labels[mask] = mode(y[mask], keepdims=True)[0][0]
            
    accuracies.append(accuracy_score(y, labels))
    precisions.append(precision_score(y, labels, average='macro', zero_division=0))
    recalls.append(recall_score(y, labels, average='macro', zero_division=0))
    f1_scores.append(f1_score(y, labels, average='macro', zero_division=0))

# pomocnicza do wykresów metryk

def plot_metric(k_vals, metric_vals, metric_name, color, marker, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, metric_vals, marker=marker, color=color, linewidth=2, markersize=8)
    plt.title(f'Wpływ liczby klastrów (k) na {metric_name}', fontsize=14)
    plt.xlabel('Liczba klastrów (k)', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(k_vals)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close() 

plot_metric(k_values, accuracies, 'Accuracy', '#4c72b0', 'o', 'accuracy_vs_k.png')
plot_metric(k_values, precisions, 'Precision (macro)', '#55a868', 's', 'precision_vs_k.png')
plot_metric(k_values, recalls, 'Recall (macro)', '#dd8452', '^', 'recall_vs_k.png')
plot_metric(k_values, f1_scores, 'F1-Score (macro)', '#c44e52', 'd', 'f1_score_vs_k.png')
print("4 wykresy metryk wygenerowane")