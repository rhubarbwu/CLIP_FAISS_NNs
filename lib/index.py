def get_image_index_filename(dataset_name, n_components, n_trees):
    return "indexes/image_{}_{}_components_{}_trees.index".format(
        dataset_name, n_components, n_trees)


def get_text_index_filename(dataset_name, n_components, n_trees):
    return "indexes/text_{}_{}_components_{}_trees.index".format(
        dataset_name, n_components, n_trees)


def build_image_index(
        dataset_name,
        n_components,
        n_trees,
        text_list_file,
        model_selection="ViT-B/32",  # RN50 or ViT-B/32
        algorithm="annoy",  # or faiss
        cuda=False):
    print(
        "Image index for {} components and {} trees, using {} does not exist. Rebuilding."
        .format(n_components, n_trees, algorithm))

    # Load the dataset and build map.
    import torchvision
    from utils import load_dataset
    dataset = load_dataset(dataset_name)

    # Load the desired model.
    import torch
    device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"
    import clip
    model, preprocess = clip.load(model_selection, device)

    # Initialize and populate index.
    from annoy import AnnoyIndex
    index = AnnoyIndex(n_components, "angular")
    index_name = get_image_index_filename(dataset_name, n_components, n_trees)
    index.on_disk_build(index_name)

    def encode_image(i):
        img, _ = dataset[i]
        img_input = preprocess(img).unsqueeze(0).to(device)
        return model.encode_image(img_input)

    from time import time
    for i in range(len(dataset.imgs)):
        start_time = time()
        index.add_item(i, encode_image(i)[0])
        print("Encoding image {} took {} seconds.".format(
            i,
            time() - start_time))

    index.build(n_trees)


def build_text_index(
        dataset_name,
        n_components,
        n_trees,
        text_list_file,
        model_selection="RN50",  # or ViT-B/32
        algorithm="annoy",  # or faiss
        cuda=False):

    print(
        "Text index for {} components and {} trees, using {} does not exist. Rebuilding."
        .format(n_components, n_trees, algorithm))

    # Load the dataset and build map.
    import torchvision
    from utils import load_dataset
    dataset = load_dataset(dataset_name)
    from utils import build_text_id_to_value_map
    text_values, text_id_to_value_map = build_text_id_to_value_map(
        dataset.classes, dataset_name, text_list_file)

    # Load the desired model.
    import torch
    device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"
    import clip
    model, preprocess = clip.load(model_selection, device)

    # Tokenize and encode the label texts.
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in text_values]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # Build NNs Index.
    c, f = text_features.shape
    from annoy import AnnoyIndex
    index = AnnoyIndex(f, "angular")
    index_name = filename_function(dataset_name, f, n_trees)
    index.on_disk_build(index_name)
    for i in range(c):
        index.add_item(i, features[i])
    index.build(n_trees)

    # Serialize the text_id->textname map
    import pickle
    from utils import get_text_map_filename
    serial_name = get_text_map_filename(dataset_name)
    with open(serial_name, 'wb') as handle:
        pickle.dump(text_id_to_value_map,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    ## Sanity check.
    with open(serial_name, 'rb') as handle:
        unpickled = pickle.load(handle)
    assert (text_id_to_value_map == unpickled)


if __name__ == "__main__":
    # build_text_index("tiny-imagenet-200", 1024, 5, None)
    build_image_index("tiny-imagenet-200", 512, 5, None)
