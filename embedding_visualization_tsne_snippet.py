import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tqdm


np.random.seed(596)

color_strings = ['red', 'orange', 'green', 'blue', 'purple', 'saddlebrown']

random_queries = np.random.choice(2000, size=len(color_strings), replace=False)

print(random_queries)

# Create a directory for plots
plots_dir = os.path.join("__plots", "tsne")

os.makedirs(plots_dir, exist_ok=True)

for i in tqdm.tqdm(range(30)):

    batch_embeds_queries = torch.load(str(i) + '_queries.pt', map_location='cpu').T
    batch_embeds_positives = torch.load(str(i) + '_positive.pt', map_location='cpu').T
    batch_embeds_pools = torch.load(str(i) + '_pools.pt', map_location='cpu').T
    
    batch_query_members = torch.load(str(i) + '_batch_members.pt', map_location='cpu')
    batch_query_members = batch_query_members[0]
    batch_nidxs = torch.load(str(i) + '_nidxs.pt', map_location='cpu')
    batch_idxs2images = torch.load(str(i) + '_idxs2images.pt', map_location='cpu')
    
    # Apply t-SNE
    print("Running t-SNE...")
    queries_tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(batch_embeds_queries)
    positives_tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(batch_embeds_positives)
    pools_tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(batch_embeds_pools)

    # Plot
    print("Plotting started...")
    plt.figure(figsize=(10, 10))
  
    # Hard negatives
    for j in range(len(pools_tsne)):
        for cn, q in enumerate(random_queries):
            if batch_idxs2images[j] in batch_nidxs[q]:
                plt.plot(pools_tsne[j, 0], pools_tsne[j, 1], marker='x', color=color_strings[cn])

    for cn, a in enumerate(random_queries):
        # Queries
        plt.plot(queries_tsne[a, 0], queries_tsne[a, 1], marker='o', color=color_strings[cn])
        
        # Positives
        plt.plot(positives_tsne[a, 0], positives_tsne[a, 1], marker="^", color=color_strings[cn])

    plt.savefig(os.path.join(plots_dir, str(i) + '_tsne.png'))
    plt.close()
