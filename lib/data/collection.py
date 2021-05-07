# collection of datasets/repositories available
collection_images = []
collection_text = []


def update_collection_images(dataset_name, dataset_path):

    refresh_collection_images()

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

    refresh_collection_text()

    if (dataset_name, dataset_path) in collection_text:
        return

    collection_text.append((dataset_name, dataset_path))
    lines = []
    for (dataset_name, dataset_path) in collection_text:
        lines.append("{} {}\n".format(dataset_name, dataset_path))

    with open("collection_text.txt", "w") as f:
        f.writelines(lines)
    f.close()


def refresh_collection_images():
    open("collection_images.txt", 'a').close()
    with open("collection_images.txt") as f:
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


def refresh_collection_text():
    open("collection_text.txt", 'a').close()
    with open("collection_text.txt") as f:
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