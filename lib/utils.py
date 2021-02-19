def get_class_map_special(classes, class_list_file, first_phrase=False):
    class_map = dict()
    words_txt = open("data/{}/words.txt".format(class_list_file), "r")
    lines = words_txt.readlines()
    for line in lines:
        code, text = line.split("	")
        text = text.strip("\n")
        class_map[code] = text.split(", ")[0] if first_phrase else text

    clean_class_map = dict()
    for c in classes:
        if c in class_map:
            clean_class_map[c] = class_map[c]

    classes_text = []
    for class_name in clean_class_map.values():
        class_name = class_name.replace("_", " ")
        classes_text.append(class_name)

    return classes_text, clean_class_map


def build_class_id_to_name_map(classes, dataset_name, class_list_file=None):
    if class_list_file is not None:
        return get_class_map_special(classes, class_list_file)

    import json
    json.loads
    with open('imagenet_class_index.json', 'r') as myfile:
        data = myfile.read()
    class_map = json.loads(data)

    classes_text = []
    for class_name in class_map.values():
        class_name = class_name[1].replace("_", " ")
        classes_text.append(class_name)

    return classes_text, class_map


def get_class_map_filename(dataset_name):
    return "maps/class_id_to_name_{}.pickle".format(dataset_name)


def get_class_index_filename(dataset_name, num_of_trees):
    return "indexes/text_{}_{}_trees.ann".format(dataset_name, num_of_trees)


def load_dataset(dataset_name):
    import torchvision
    return torchvision.datasets.ImageFolder(
        "data/{}/train".format(dataset_name))
