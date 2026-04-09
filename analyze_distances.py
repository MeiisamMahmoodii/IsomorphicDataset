import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
CSV_PATH = "alignment_dataset.csv"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOAD DATASET ---
df = pd.read_csv(CSV_PATH)

# --- LOAD MODEL ---
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
model.eval()

# --- EMBEDDING FUNCTION ---
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled.cpu().numpy()

# --- COMPUTE EMBEDDINGS ---
print("Computing embeddings for all sentences...")
orig_embeds = []
rewrite_embeds = []

for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch = df.iloc[i:i+BATCH_SIZE]
    orig_embeds.append(get_embeddings(batch['original_text'].tolist()))
    rewrite_embeds.append(get_embeddings(batch['rewritten_text'].tolist()))

orig_embeds = np.vstack(orig_embeds)
rewrite_embeds = np.vstack(rewrite_embeds)

# --- COMPUTE DISTANCES ---
print("\nComputing Wasserstein (L2) distances...")
w_dist = np.linalg.norm(orig_embeds - rewrite_embeds, axis=1)

# --- ANALYZE DISTRIBUTION ---
print("\n=== DISTANCE STATISTICS ===")
print(f"Min:      {w_dist.min():.4f}")
print(f"Max:      {w_dist.max():.4f}")
print(f"Mean:     {w_dist.mean():.4f}")
print(f"Median:   {np.median(w_dist):.4f}")
print(f"Std Dev:  {w_dist.std():.4f}")
print(f"Q25:      {np.percentile(w_dist, 25):.4f}")
print(f"Q75:      {np.percentile(w_dist, 75):.4f}")

print("\n=== PERCENTILE BREAKDOWN ===")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(w_dist, p)
    count = (w_dist < val).sum()
    print(f"P{p:2d}: {val:.4f} ({count:4d} pairs below)")

# --- SUGGEST THRESHOLD ---
print("\n=== THRESHOLD SUGGESTIONS ===")
for threshold in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
    kept = (w_dist < threshold).sum()
    pct = 100 * kept / len(w_dist)
    print(f"Threshold {threshold:.1f}: {kept:5d} pairs ({pct:5.1f}%)")
