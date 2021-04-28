def get_image_index_filename(dataset_name, n_components):
    return "indexes_images/{}_{}.index".format(dataset_name, n_components)


def get_txt_index_filename(dataset_name, n_components):
    return "indexes_text/{}_{}.index".format(dataset_name, n_components)


def build_txt_list(classes=None,
                   txt_list_file="data/imagenet_class_index.json"):
    import json

    # open txt file for reading
    with open(txt_list_file, "r") as myfile:
        data = myfile.read()
    txt_map = json.loads(data)

    # build txt list
    txt = []
    for txt_value in txt_map.values():
        txt_value = txt_value[1].replace("_", " ")
        txt.append(txt_value)

    # remove any unwanted txt
    if classes is not None:
        for i in range(len(txt) - 1, -1, -1):
            if len(classes) == 0 or classes[-1] != txt[i][0]:
                txt.pop(i)
            elif len(classes) > 0:
                classes.pop()

    return txt
