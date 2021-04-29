from ..hparams import model_selection

import os, torch

if "LOCAL_CLIP" in os.environ:
    from CLIP.clip import clip
else:
    import clip

# Load the desired model.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_selection, device)
