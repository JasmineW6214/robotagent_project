#!/usr/bin/env python3
"""Quick script to check actual prefix length during inference."""

import sys
sys.path.insert(0, '/home/jrobot/project/lerobot/src')

import torch
from pathlib import Path

print("Loading policy...", flush=True)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

CHECKPOINT = Path('/home/jrobot/project/lerobot/outputs/smolvla_bimanual_20260103_200201/checkpoints/080000/pretrained_model')
policy = SmolVLAPolicy.from_pretrained(str(CHECKPOINT))
policy.eval()
policy.to('cuda')

print("Loading preprocessor...", flush=True)
dataset_path = Path('/home/jrobot/project/lerobot/datasets_bimanuel/multitasks')
dataset_metadata = LeRobotDatasetMetadata(repo_id='multitasks', root=str(dataset_path))

preprocessor, _ = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=str(CHECKPOINT),
    dataset_stats=dataset_metadata.stats,
    preprocessor_overrides={'device_processor': {'device': 'cuda'}},
)

# Create fake observation with 3 cameras at 480x640 (like saved images)
def fake_img():
    return torch.rand(1, 3, 480, 640)

observation = {
    'observation.state': torch.zeros(1, 12).cuda(),
    'observation.images.camera1': fake_img().cuda(),
    'observation.images.camera2': fake_img().cuda(),
    'observation.images.camera3': fake_img().cuda(),
    'task': 'test task',
}

print("Preprocessing...", flush=True)
preprocessed = preprocessor(observation)

print('=== After preprocessing ===', flush=True)
for k, v in preprocessed.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: {v.shape}', flush=True)
    else:
        print(f'  {k}: {type(v).__name__}', flush=True)

# Hook into embed_prefix to capture prefix length
original_embed_prefix = policy.model.embed_prefix

captured_prefix_shape = []
captured_images = []

def capturing_embed_prefix(images, img_masks, *args, **kwargs):
    captured_images.append([img.shape for img in images])
    result = original_embed_prefix(images, img_masks, *args, **kwargs)
    prefix_embs = result[0]
    captured_prefix_shape.append(prefix_embs.shape)
    return result

policy.model.embed_prefix = capturing_embed_prefix

print("\nRunning inference...", flush=True)
policy.reset()

with torch.no_grad():
    action = policy.select_action(preprocessed)

print(f'\n=== Prefix shape during inference ===', flush=True)
for i, shape in enumerate(captured_prefix_shape):
    print(f'Prefix embedding shape: {shape}', flush=True)
    print(f'Total prefix tokens: {shape[1]}', flush=True)

print(f'\n=== Images passed to embed_prefix ===', flush=True)
for img_shapes in captured_images:
    print(f'Number of images: {len(img_shapes)}', flush=True)
    for j, s in enumerate(img_shapes):
        print(f'  Image {j}: {s}', flush=True)
