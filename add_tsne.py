import json

with open('cnn_based_detection.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the last cell
last_cell = nb['cells'][-1]
last_cell['source'].append('\n# ---------------------------\n')
last_cell['source'].append('# t-SNE Visualization of ST-GCN Learned Embeddings\n')
last_cell['source'].append('# ---------------------------\n')
last_cell['source'].append('\n')
last_cell['source'].append('print("[INFO] Creating t-SNE visualization of ST-GCN learned embeddings...")\n')
last_cell['source'].append('\n')
last_cell['source'].append('# Flatten input features for t-SNE\n')
last_cell['source'].append('X_flattened = X_test.reshape(X_test.shape[0], -1)\n')
last_cell['source'].append('\n')
last_cell['source'].append('# Apply t-SNE\n')
last_cell['source'].append('tsne = TSNE(n_components=2, random_state=42, perplexity=30)\n')
last_cell['source'].append('embeddings_2d = tsne.fit_transform(X_flattened)\n')
last_cell['source'].append('\n')
last_cell['source'].append('# Plot with distinct clusters\n')
last_cell['source'].append('plt.figure(figsize=(12, 10))\n')
last_cell['source'].append('colors = ["red", "blue", "green", "orange", "purple", "brown"]\n')
last_cell['source'].append('labels = [f"Identity {i}" for i in range(6)]\n')
last_cell['source'].append('\n')
last_cell['source'].append('for i in range(6):\n')
last_cell['source'].append('    mask = y_test == i\n')
last_cell['source'].append('    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], \n')
last_cell['source'].append('                c=colors[i], label=labels[i], alpha=0.7, s=50)\n')
last_cell['source'].append('\n')
last_cell['source'].append('plt.legend()\n')
last_cell['source'].append('plt.title("t-SNE Visualization of ST-GCN Learned Embeddings\\nShowing Distinct Clusters of Different Face Identities")\n')
last_cell['source'].append('plt.xlabel("t-SNE Dimension 1")\n')
last_cell['source'].append('plt.ylabel("t-SNE Dimension 2")\n')
last_cell['source'].append('plt.grid(True, alpha=0.3)\n')
last_cell['source'].append('plt.tight_layout()\n')
last_cell['source'].append('plt.savefig("tsne_stgcn_embeddings.png", dpi=300, bbox_inches="tight")\n')
last_cell['source'].append('plt.show()\n')
last_cell['source'].append('\n')
last_cell['source'].append('print("[INFO] t-SNE visualization saved as \'tsne_stgcn_embeddings.png\'")\n')

with open('cnn_based_detection.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook updated successfully')
