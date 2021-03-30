def get_image_index_filename(dataset_name, n_components, label="untitled"):
    return "indexes/image_{}_{}_{}.index".format(dataset_name, label,
                                                 n_components)


def get_text_index_filename(dataset_name, n_components):
    return "indexes_text/text_{}_{}.index".format(dataset_name, n_components)


def get_text_map_filename(dataset_name):
    return "maps/{}.pickle".format(dataset_name)


def build_text_id_to_value_map(classes=None,
                               text_list_file="data/imagenet_class_index.json"
                               ):
    import json

    with open(text_list_file, "r") as myfile:
        data = myfile.read()
    text_map = json.loads(data)

    text = []
    for text_value in text_map.values():
        text_value = text_value[1].replace("_", " ")
        text.append(text_value)

    text_list = [None] * len(text)
    for text_id in text_map:
        text_list[int(text_id)] = text_map[text_id]

    if classes is not None:
        for i in range(len(text) - 1, -1, -1):
            if len(classes) == 0 or classes[-1] != text_list[i][0]:
                text.pop(i)
                text_list.pop(i)
            elif len(classes) > 0:
                classes.pop()

    return text, text_list
