from .model import *
from PIL import Image


def encode_img(dataset, i):
    img, idx = dataset[i]
    img_input = preprocess(img).unsqueeze(0).to(device)
    return model.encode_image(img_input).detach().cpu().numpy().astype(
        'float32')


def encode_img_by_path(path):
    img_input = preprocess(Image.open(path)).unsqueeze(0).to(device)
    return model.encode_image(img_input).detach().cpu().numpy().astype(
        'float32')


def encode_txt(text_values):
    text_inputs = clip.tokenize(text_values).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features.detach().cpu().numpy().astype('float32')
