def query_image_to_text(dataset_name, idx, n_components, n_neighbours, n_trees):
    idx = idx % 100  # TOY: remove in production

    # Load the image AnnoyIndex and get the encoding of the image for idx.
    from annoy import AnnoyIndex
    image_index = AnnoyIndex(n_components, "angular")
    from index import get_image_index_filename
    image_index.load(
        get_image_index_filename(dataset_name, n_components, n_trees))
    img_features = image_index.get_item_vector(idx)

    # Load the text AnnoyIndex.
    from index import get_text_index_filename
    text_index = AnnoyIndex(n_components, "angular")
    text_index.load(get_text_index_filename(dataset_name, n_components,
                                            n_trees))

    # For the image encoding, find the nearest neighbour text encodings.
    top_k = text_index.get_nns_by_vector(img_features, n_neighbours)

    return top_k


def query_text_to_image(dataset_name,
                        n_components,
                        n_neighbours,
                        n_trees,
                        text,
                        model_selection="RN50",
                        cuda=False):

    # Load the desired model.
    import torch
    device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"
    import clip
    model, preprocess = clip.load(model_selection, device)
    text_input = clip.tokenize([text]).to(device)
    text_features = model.encode_text(text_input)

    # Load the image AnnoyIndex
    from annoy import AnnoyIndex
    image_index = AnnoyIndex(n_components, "angular")
    from index import get_image_index_filename
    image_index.load(
        get_image_index_filename(dataset_name, n_components, n_trees))

    # For the text encoding, find the nearest neighbour image encodings.
    top_k = image_index.get_nns_by_vector(text_features[0], n_neighbours)

    return top_k


def query_image_to_image():
    return
