#!/usr/bin/env python3
"""Check how image tokens are distributed across cameras."""

import sys
sys.path.insert(0, '/home/jrobot/project/lerobot/src')

import torch
from pathlib import Path

print("Loading policy...", flush=True)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CHECKPOINT = Path('/home/jrobot/project/lerobot/outputs/smolvla_bimanual_20260103_200201/checkpoints/080000/pretrained_model')
policy = SmolVLAPolicy.from_pretrained(str(CHECKPOINT))
policy.eval()
policy.to('cuda')

# Check the model's embed_prefix to understand token structure
# Let's trace through with a simple test

print("\n=== Analyzing embed_prefix internals ===", flush=True)

# Create 3 fake 512x512 images
images = [
    torch.randn(1, 3, 512, 512).cuda() for _ in range(3)
]
img_masks = [torch.tensor([True]).cuda() for _ in range(3)]

# Tokenize a test task
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
task = "Use left arm to pick up the yogurt bottle"
tokens = tokenizer(task, return_tensors="pt", padding="max_length", max_length=48)
lang_tokens = tokens["input_ids"].cuda()
lang_masks = tokens["attention_mask"].cuda()

# State
state = torch.zeros(1, 12).cuda()

print("Calling embed_prefix...", flush=True)

# Let's trace through embed_prefix manually
model = policy.model

# Step through embed_prefix logic
embs_per_image = []
for i, (img, img_mask) in enumerate(zip(images, img_masks)):
    # Pass through vision encoder
    img_emb = model.get_image_embedding(img)
    print(f"  Image {i} embedding shape: {img_emb.shape}", flush=True)
    embs_per_image.append(img_emb.shape)

# Call actual embed_prefix
prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
    images, img_masks, lang_tokens, lang_masks, state=state
)

print(f"\nFinal prefix shape: {prefix_embs.shape}", flush=True)

# Calculate token breakdown
total_tokens = prefix_embs.shape[1]
lang_tokens_count = 48  # From config
state_tokens_count = 1
image_tokens_total = total_tokens - lang_tokens_count - state_tokens_count

print(f"\n=== Token breakdown ===", flush=True)
print(f"Total prefix tokens: {total_tokens}", flush=True)
print(f"Language tokens: ~{lang_tokens_count}", flush=True)
print(f"State tokens: ~{state_tokens_count}", flush=True)
print(f"Image tokens (all 3 cameras): ~{image_tokens_total}", flush=True)
print(f"Image tokens per camera: ~{image_tokens_total // 3}", flush=True)

grid_size = int((image_tokens_total // 3) ** 0.5)
print(f"Grid size per camera: ~{grid_size}x{grid_size}", flush=True)

# Estimate token boundaries
print(f"\n=== Estimated token boundaries ===", flush=True)
img_per_cam = image_tokens_total // 3
print(f"Camera 1 (head):       tokens [0 : {img_per_cam}]", flush=True)
print(f"Camera 2 (left_wrist): tokens [{img_per_cam} : {2*img_per_cam}]", flush=True)
print(f"Camera 3 (right_wrist): tokens [{2*img_per_cam} : {3*img_per_cam}]", flush=True)
print(f"Language:              tokens [{3*img_per_cam} : {3*img_per_cam + lang_tokens_count}]", flush=True)
print(f"State:                 tokens [{3*img_per_cam + lang_tokens_count} : {total_tokens}]", flush=True)
