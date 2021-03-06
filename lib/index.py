from annoy import AnnoyIndex
from hparams import *
from time import time

import clip, faiss, pickle, torch, torchvision

# Load the desired model.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_selection, device)


def get_image_index_filename(dataset_name, n_components, n_trees=None):
    return "indexes/image_{}_{}_components{}.index".format(
        dataset_name, n_components,
        "" if n_trees is None else "_{}_trees".format(n_trees))


def get_text_index_filename(dataset_name, n_components, n_trees=None):
    return "indexes/text_{}_{}_components{}.index".format(
        dataset_name, n_components,
        "" if n_trees is None else "_{}_trees".format(n_trees))


def encode_image(dataset, i):
    img, idx = dataset[i]
    img_input = preprocess(img).unsqueeze(0).to(device)
    return model.encode_image(img_input).detach().numpy()


def build_image_index_annoy(
        dataset_name,
        dataset_path,
        n_components=n_components,
        n_trees=n_trees,
        text_list_file=text_list_file,
        model_selection=model_selection,  # RN50 or ViT-B/32
        image_limit=None,
        verbose=False):

    print("Building new image ANNOY index for {} components with {}...".format(
        n_components, n_trees))

    # Load the dataset and build map.
    dataset = torchvision.datasets.ImageFolder(dataset_path)

    # Filename of the index to load/write to.
    index_filename = get_image_index_filename(dataset_name, n_components,
                                              n_trees)

    # Initialize and populate index.
    index = AnnoyIndex(n_components, "angular")
    index_name = get_image_index_filename(dataset_name, n_components, n_trees)
    index.on_disk_build(index_name)

    # Use image limit.
    if image_limit is None:
        image_limit = len(dataset.imgs)

    # Add each image's encoding to the index.
    for i in range(image_limit):
        start_time = time()
        index.add_item(i, encode_image(dataset, i)[0])
        print("Encoding image {} took {} seconds.".format(
            i,
            time() - start_time))

    index.build(n_trees)

    return


def build_image_index_faiss(
        dataset_name,
        dataset_path,
        n_components=n_components,
        text_list_file=text_list_file,
        model_selection=model_selection,  # RN50 or ViT-B/32
        image_limit=None,
        append=False,
        verbose=False):

    if append:
        print("Adding new images to existing FAISS index for {} components...".
              format(n_components))
    else:
        print("Building new image FAISS index for {} components...".format(
            n_components))

    # Load the dataset and build map.
    dataset = torchvision.datasets.ImageFolder(dataset_path)

    # Filename of the index to load/write to.
    index_filename = get_image_index_filename(dataset_name, n_components)

    # Read existing index if append, or start fresh index.
    if append:
        index = faiss.read_index(index_filename)
    else:
        index = faiss.IndexFlatL2(n_components)

    # Use image limit.
    if image_limit is None:
        image_limit = len(dataset.imgs)

    # Add each image's encoding to the index.
    for i in range(image_limit):
        if verbose:
            start_time = time()
        index.add(encode_image(dataset, i))
        if verbose:
            print("  Encoding image {} took {} seconds.".format(
                i,
                time() - start_time))

    # Write to index.
    faiss.write_index(index, index_filename)


def build_text_index_annoy(dataset_name,
                           n_components=n_components,
                           classes=None,
                           model_selection=model_selection):

    print("Building new text ANNOY index for {} components with {} trees...".
          format(n_components, n_trees))

    # Build text list and map.
    from utils import build_text_id_to_value_map
    text_values, text_id_to_value_map = build_text_id_to_value_map(classes)

    # Tokenize and encode the label texts.
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of {c}") for c in text_values]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # Build NNs Index.
    c, f = text_features.shape
    from annoy import AnnoyIndex
    index = AnnoyIndex(f, "angular")
    index_name = get_text_index_filename(dataset_name, n_components, n_trees)
    index.on_disk_build(index_name)
    for i in range(c):
        index.add_item(i, text_features[i])
    index.build(5)


def build_text_index_faiss(dataset_name,
                           n_components=n_components,
                           classes=None,
                           model_selection=model_selection):

    print("Building new text FAISS index for {} components...".format(
        n_components))

    # Build text list and map.
    from utils import build_text_id_to_value_map
    text_values, text_id_to_value_map = build_text_id_to_value_map(classes)

    # Tokenize and encode the label texts.
    text_inputs = clip.tokenize(text_values).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # Filename of the index to load/write to.
    index_filename = get_text_index_filename(dataset_name, n_components)

    # Build NNs Index.
    n, c = text_features.shape
    index = faiss.IndexFlatL2(n_components)
    index.add(text_features.detach().numpy())
    faiss.write_index(index, index_filename)


if __name__ == "__main__":
    start_time = time()

    build_text_index_func = build_text_index_annoy
    build_text_index_func(dataset_name, classes=None)

    build_image_index_func = build_image_index_annoy
    build_image_index_func(dataset_name, dataset_path, image_limit=10000)
    print("Indexing completed in {}.".format(time() - start_time))
