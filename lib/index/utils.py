def get_image_index_filename(dataset_name, n_components):
    return "indexes_images/{}_{}.index".format(dataset_name, n_components)


def get_text_index_filename(dataset_name, n_components):
    return "indexes_text/{}_{}.index".format(dataset_name, n_components)


def build_text_list(classes=None,
                    text_list_file="data/imagenet_class_index.json"):
    import json

    # open text file for reading
    with open(text_list_file, "r") as myfile:
        data = myfile.read()
    text_map = json.loads(data)

    # build text list
    text = []
    for text_value in text_map.values():
        text_value = text_value[1].replace("_", " ")
        text.append(text_value)

    # remove any unwanted text
    if classes is not None:
        for i in range(len(text) - 1, -1, -1):
            if len(classes) == 0 or classes[-1] != text[i][0]:
                text.pop(i)
            elif len(classes) > 0:
                classes.pop()

    return text
