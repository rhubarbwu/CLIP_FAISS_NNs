def query_image_to_text_one(
        dataset_name,
        img,
        num_of_neighbours,
        num_of_trees=10,
        model_selection="ViT-B/32",  # or RN50
        algorithm="annoy",
        class_list_file="tiny-imagenet-200",
        cuda=False):
    # Initialize PyTorch
    import torch, torchvision
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Deserialize the class ID->name map.
    import pickle
    from utils import get_class_map_filename
    serial_name = get_class_map_filename(dataset_name)
    with open(serial_name, 'rb') as handle:
        unpickled = pickle.load(handle)

    # Start timing.
    import time
    start_time = time.time()

    # Load the desired model.
    import clip
    model, preprocess = clip.load(model_selection, device)

    # Preprocess and encode the image.
    img_input = preprocess(img).unsqueeze(0).to(device)
    img_features = model.encode_image(img_input)

    # Initialize and load the text index.
    from annoy import AnnoyIndex
    f = img_features.shape[1]
    class_index = AnnoyIndex(f, 'angular')
    from utils import get_class_index_filename
    class_index_filename = get_class_index_filename(dataset_name, num_of_trees)

    # Pause time.
    elapsed_time = time.time() - start_time

    # Rebuild index if necessary.
    import os
    if not os.path.isfile(class_index_filename):
        from build_class_index import build_class_index
        build_class_index(dataset_name, num_of_trees, class_list_file,
                          model_selection, algorithm, cuda)
    class_index.load(class_index_filename)

    # Return k nearest neighbours and add the elapsed time.
    start_time = time.time()
    top_k = class_index.get_nns_by_vector(img_features[0], num_of_neighbours)
    elapsed_time += time.time() - start_time

    return top_k, elapsed_time