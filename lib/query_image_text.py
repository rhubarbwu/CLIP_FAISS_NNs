def query_image_to_text_one(
        dataset_name,
        img,
        num_of_components,
        num_of_neighbours,
        num_of_trees,
        model_selection="RN50",  # or ViT-B/32
        algorithm="annoy",
        text_list_file="tiny-imagenet-200",
        cuda=False):
    # Initialize PyTorch
    import torch, torchvision
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build index if it doesn't exist.
    from utils import get_text_map_filename
    serial_name = get_text_map_filename(dataset_name)

    # Deserialize the text ID->name map.
    import pickle
    with open(serial_name, 'rb') as handle:
        unpickled = pickle.load(handle)

    # Start timing.
    import time
    start_time = time.time()

    # Load the desired model.
    import clip
    model, preprocess = clip.load(model_selection, device, jit=True)

    # Preprocess and encode the image.
    img_input = preprocess(img).unsqueeze(0).to(device)
    img_features = model.encode_image(img_input)

    # Pause time.
    elapsed_time = time.time() - start_time

    # Initialize and load the text index.
    from annoy import AnnoyIndex
    f = img_features.shape[1]
    text_index = AnnoyIndex(f, "angular")
    from index import get_text_index_filename
    text_index_filename = get_text_index_filename(dataset_name,
                                                  num_of_components,
                                                  num_of_trees)
    text_index.load(text_index_filename)

    # Return k nearest neighbours and add the elapsed time.
    start_time = time.time()
    top_k = text_index.get_nns_by_vector(img_features[0], num_of_neighbours)
    elapsed_time += time.time() - start_time

    return top_k, elapsed_time