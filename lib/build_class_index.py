def build_class_index(
        dataset_name,
        num_of_trees,
        class_list_file,
        model_selection="ViT-B/32",  # or RN50
        algorithm="annoy",  # or faiss
        cuda=False):

    print("Index for {} using {} does not exist. Rebuilding.".format(
        num_of_trees, algorithm))

    # Load the dataset and build map.
    import torchvision
    from utils import load_dataset
    dataset = load_dataset(dataset_name)
    from utils import build_class_id_to_name_map
    class_names, class_id_to_name_map = build_class_id_to_name_map(
        dataset.classes, dataset_name, class_list_file)

    # Load the desired model.
    import torch
    device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"
    import clip
    model, preprocess = clip.load(model_selection, device)

    # Tokenize and encode the label texts.
    class_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    with torch.no_grad():
        class_features = model.encode_text(class_inputs)

    # Build NNs Index
    if algorithm == "annoy":
        build_annoy_index(class_features, dataset_name, num_of_trees)
    elif algorithm == "faiss":
        build_faiss_index(class_features)

    # Serialize the class_id->classname map
    import pickle
    from utils import get_class_map_filename
    serial_name = get_class_map_filename(dataset_name)
    with open(serial_name, 'wb') as handle:
        pickle.dump(class_id_to_name_map,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    ## Sanity check.
    with open(serial_name, 'rb') as handle:
        unpickled = pickle.load(handle)
    assert (class_id_to_name_map == unpickled)


def build_annoy_index(class_features, dataset_name, num_of_trees):
    c, f = class_features.shape

    # Initialize and populate index.
    from annoy import AnnoyIndex
    class_index = AnnoyIndex(f, 'angular')
    for i in range(c):
        class_index.add_item(i, class_features[i])
    class_index.build(num_of_trees)

    # Save the index to file.
    from utils import get_class_index_filename
    index_name = get_class_index_filename(dataset_name, num_of_trees)
    class_index.save(index_name)


def build_faiss_index(class_features):
    print("Unimplemented!")
    exit()
