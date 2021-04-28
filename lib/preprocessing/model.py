from ..hparams import model_selection

import clip, torch

# Load the desired model.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_selection, device)
