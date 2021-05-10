# collection of datasets/repositories available
collection_images = []
collection_text = []


def update_collection_images(dataset_name, dataset_path, collection_file):

    refresh_collection_images(collection_file)

    if (dataset_name, dataset_path) in collection_images:
        return

    collection_images.append((dataset_name, dataset_path))
    lines = []
    for (dataset_name, dataset_path) in collection_images:
        lines.append("{} {}\n".format(dataset_name, dataset_path))

    with open(collection_file, "w") as f:
        f.writelines(lines)
    f.close()


def update_collection_text(dataset_name, dataset_path, collection_file):

    refresh_collection_text(collection_file)

    if (dataset_name, dataset_path) in collection_text:
        return

    collection_text.append((dataset_name, dataset_path))
    lines = []
    for (dataset_name, dataset_path) in collection_text:
        lines.append("{} {}\n".format(dataset_name, dataset_path))

    with open(collection_file, "w") as f:
        f.writelines(lines)
    f.close()


def refresh_collection_images(collection_file):
    open(collection_file, 'a').close()
    with open(collection_file) as f:
        lines = f.readlines()
        f.close()
    collection = []
    for line in lines:
        split = line.split()
        if len(split) == 2:
            collection.append((split[0], split[1]))

    global collection_images
    collection_images = collection
    return collection


def refresh_collection_text(collection_file):
    open(collection_file, 'a').close()
    with open(collection_file) as f:
        lines = f.readlines()
        f.close()
    collection = []
    for line in lines:
        split = line.split()
        if len(split) == 2:
            collection.append((split[0], split[1]))

    global collection_text
    collection_text = collection
    return collection