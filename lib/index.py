def get_image_index_filename(dataset_name, num_of_components, num_of_trees):
    return "indexes/image_{}_{}_components_{}_trees.index".format(
        dataset_name, num_of_components, num_of_trees)


def get_text_index_filename(dataset_name, num_of_components, num_of_trees):
    return "indexes/text_{}_{}_components_{}_trees.index".format(
        dataset_name, num_of_components, num_of_trees)


def build_annoy_index(features,
                      dataset_name,
                      num_of_trees,
                      filename_function=get_image_index_filename,
                      distance="angular"):
    c, f = features.shape

    # Initialize and populate index.
    from annoy import AnnoyIndex
    index = AnnoyIndex(f, 'angular')
    index_name = filename_function(dataset_name, f, num_of_trees)
    index.on_disk_build(index_name)
    for i in range(c):
        index.add_item(i, features[i])
    index.build(num_of_trees)


def build_image_index(
        dataset_name,
        num_of_components,
        num_of_trees,
        text_list_file,
        model_selection="RN50",  # or ViT-B/32
        algorithm="annoy",  # or faiss
        cuda=False):
    print(
        "Image index for {} components and {} trees, using {} does not exist. Rebuilding."
        .format(num_of_components, num_of_trees, algorithm))

    # Load the dataset and build map.
    import torchvision
    from utils import load_dataset
    dataset = load_dataset(dataset_name)

    # Load the desired model.
    import torch
    device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"
    import clip
    model, preprocess = clip.load(model_selection, device)

    from time import time
    images_features = None
    for i in range(len(dataset)):  # make this len(dataset)
        img, _ = dataset[i]
        img_input = preprocess(img).unsqueeze(0).to(device)
        start_time = time()
        features = model.encode_image(img_input)
        images_features = features if images_features is None else torch.cat(
            (images_features, features))
        print("Encoding image {} took {} seconds.".format(
            i,
            time() - start_time))

    build_annoy_index(images_features, dataset_name, num_of_trees)


def build_text_index(
        dataset_name,
        num_of_components,
        num_of_trees,
        text_list_file,
        model_selection="RN50",  # or ViT-B/32
        algorithm="annoy",  # or faiss
        cuda=False):

    print(
        "Text index for {} components and {} trees, using {} does not exist. Rebuilding."
        .format(num_of_components, num_of_trees, algorithm))

    # Load the dataset and build map.
    import torchvision
    from utils import load_dataset
    dataset = load_dataset(dataset_name)
    from utils import build_text_id_to_value_map
    text_names, text_id_to_value_map = build_text_id_to_value_map(
        dataset.classes, dataset_name, text_list_file)

    # Load the desired model.
    import torch
    device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"
    import clip
    model, preprocess = clip.load(model_selection, device)

    # Tokenize and encode the label texts.
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in text_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # Build NNs Index
    build_annoy_index(text_features,
                      dataset_name,
                      num_of_trees,
                      filename_function=get_text_index_filename)

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
