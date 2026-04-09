import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from scipy.spatial.distance import cdist

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
CSV_PATH = "alignment_dataset.csv"
OUTPUT_PATH = "alignment_dataset_with_wdist.csv"
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
        # Use mean pooling (recommended for Qwen3-Embedding-8B)
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

# --- NORMALIZE EMBEDDINGS ---
print("Normalizing embeddings...")
orig_embeds = orig_embeds / np.linalg.norm(orig_embeds, axis=1, keepdims=True)
rewrite_embeds = rewrite_embeds / np.linalg.norm(rewrite_embeds, axis=1, keepdims=True)

# --- CALCULATE WASSERSTEIN (L2) DISTANCE ---
print("Calculating Wasserstein (L2) distances...")
w_dist = np.linalg.norm(orig_embeds - rewrite_embeds, axis=1)
df['w_distance'] = w_dist

# --- FILTER BY THRESHOLD ---
THRESHOLD = 0.5
keep_mask = df['w_distance'] < THRESHOLD
kept = df[keep_mask]
print(f"Kept {kept.shape[0]} / {df.shape[0]} pairs (W < {THRESHOLD})")

# --- SAVE OUTPUT ---
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved results to {OUTPUT_PATH}")
