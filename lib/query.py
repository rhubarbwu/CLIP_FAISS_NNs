def query_image_to_text(dataset_name, idx, num_of_components, num_of_neighbours,
                        num_of_trees):
    idx = idx % 100  # TOY: remove in production

    from time import time
    start_time = time()

    from annoy import AnnoyIndex
    image_index = AnnoyIndex(num_of_components, "angular")

    from index import get_image_index_filename
    image_index.load(
        get_image_index_filename(dataset_name, num_of_components, num_of_trees))
    img_features = image_index.get_item_vector(idx)

    from index import get_text_index_filename
    text_index = AnnoyIndex(num_of_components, "angular")
    text_index.load(
        get_text_index_filename(dataset_name, num_of_components, num_of_trees))

    top_k = text_index.get_nns_by_vector(img_features, num_of_neighbours)

    print(time() - start_time)
    return top_k


def query_text_to_image():
    return


def query_image_to_image():
    return


if __name__ == "__main__":

    dataset_name = "tiny-imagenet-200"
    dataset_size = 100  # TOY: 100000 for full tiny-imagenet-200
    num_of_components = 1024
    num_of_neighbours = 5
    num_of_samples = 100
    num_of_trees = 5
    assert num_of_samples <= dataset_size

    import torchvision
    from utils import load_dataset
    dataset = load_dataset(dataset_name)

    import random
    indices = random.sample(range(dataset_size), num_of_samples)

    from time import time
    top_1_count, top_k_count = 0, 0
    total_time = 0.0
    for idx in indices:
        img, text_idx = dataset[idx]

        start_time = time()
        top_k = query_image_to_text(dataset_name, idx, num_of_components,
                                    num_of_neighbours, num_of_trees)
        query_time = time() - start_time
        total_time += query_time

        if text_idx == top_k[0]:
            top_1_count += 1
            top_k_count += 1
        elif text_idx in top_k:
            top_k_count += 1

        print(text_idx, top_k, query_time)

    print(
        "\n\t After querying {} examples in an average of {} seconds...".format(
            num_of_samples, total_time / num_of_samples), "\n\tTop 1 Accuracy:",
        top_1_count / num_of_samples,
        "\n\tTop {} Accuracy:".format(num_of_neighbours),
        top_k_count / num_of_samples)