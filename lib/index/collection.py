from lib.hparams import collection_images, collection_text


def update_collection_images(dataset_name, dataset_path):

    if (dataset_name, dataset_path) in collection_images:
        return

    collection_images.append((dataset_name, dataset_path))
    lines = []
    for (dataset_name, dataset_path) in collection_images:
        lines.append("{} {}\n".format(dataset_name, dataset_path))

    with open("collection_images.txt", "w") as f:
        f.writelines(lines)
    f.close()


def update_collection_text(dataset_name, dataset_path):

    if (dataset_name, dataset_path) in collection_text:
        return

    collection_text.append((dataset_name, dataset_path))
    lines = []
    for (dataset_name, dataset_path) in collection_text:
        lines.append("{} {}\n".format(dataset_name, dataset_path))

    with open("collection_text.txt", "w") as f:
        f.writelines(lines)
    f.close()
