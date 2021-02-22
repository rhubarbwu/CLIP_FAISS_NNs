def get_text_map_special(raw_text, text_list_file, first_phrase=False):
    text_map = dict()
    words_txt = open("data/{}/words.txt".format(text_list_file), "r")
    lines = words_txt.readlines()
    for line in lines:
        code, text_str = line.split("	")
        text_str = text_str.strip("\n")
        text_map[code] = text_str.split(", ")[0] if first_phrase else text_str

    clean_text_map = dict()
    for c in raw_text:
        if c in text_map:
            clean_text_map[c] = text_map[c]

    text = []
    for text_name in clean_text_map.values():
        text_name = text_name.replace("_", " ")
        text.append(text_name)

    return text, clean_text_map


def build_text_id_to_value_map(raw_text, dataset_name, text_list_file=None):
    if text_list_file is not None:
        return get_text_map_special(raw_text, text_list_file)

    import json
    json.loads
    with open("data/imagenet_class_index.json", "r") as myfile:
        data = myfile.read()
    text_map = json.loads(data)

    text = []
    for text_name in text_map.values():
        text_name = text_name[1].replace("_", " ")
        text.append(text_name)

    return text, text_map


def get_text_map_filename(dataset_name):
    return "maps/{}.pickle".format(dataset_name)


def load_dataset(dataset_name):
    import torchvision
    return torchvision.datasets.ImageFolder(
        "data/{}/train".format(dataset_name))
