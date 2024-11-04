import yaml
import os
import math
import csv
import random
import time
from tqdm import tqdm
import json
from utils.argUtils import CustomObject, get_yaml_loader


def scrub_data(dataSetPath):

    print("Collecting image text pairs...")
    image_text_data = {"train":[], "valid":[]}
    start = time.time()
    for split in ["train", "valid"]:

        with open(dataSetPath+f'/captions_{split}2017.json', 'r') as f:
            data = json.load(f)

        map_data = {}
        for images in tqdm(data['images']):
            map_data[images['id']] = images['coco_url']

        for annotation in tqdm(data['annotations']):
            image_text_data[split].append([map_data[annotation['image_id']], annotation['caption']])

    print(f"Time taken [Collection] = {((time.time() - start) / 60):.5f} seconds")
    print(f"Crawled data:\n\tTraining: {len(image_text_data['train'])}\n\tValidation: {len(image_text_data['valid'])}")
    return image_text_data


def process_metadata(dataSetPath):

    image_text_data = scrub_data(dataSetPath)

    return image_text_data


def limit_on_total(Metadata, data_to_write):

    filtered_data_to_write = {"train":[], "valid":[]}
    print(f"Processing filtering ...")
    start = time.time()

    for split in ["train", "valid"]:

        limit = math.ceil(Metadata.Value * len(data_to_write[split]))

        mask = random.sample(range(len(data_to_write[split])), limit)

        filtered_data_to_write[split] = [data_to_write[split][idx] for idx in tqdm(mask)]

    print(f"Time taken [Filtering] = {((time.time() - start) / 60):.5f} seconds")
    return filtered_data_to_write


def write_metadata(dataSetPath, data_to_write):

    for dataset in ["", "_train", "_valid"]:
        with open(f"{dataSetPath}/metadata{dataset}.csv", 'a', newline='') as metadata:
            writer = csv.writer(metadata)
            writer.writerow(["inputPath", "label"])
    start = time.time()

    for split in ["train", "valid"]:
        with open(f"{dataSetPath}/metadata_{split}.csv", 'a', newline='') as metadata:
            writer = csv.writer(metadata)
            writer.writerows(data_to_write[split])

    with open(f"{dataSetPath}/metadata.csv", 'a', newline='') as metadata:
        writer = csv.writer(metadata)
        writer.writerows(data_to_write["train"] + data_to_write["valid"])

    print(f"Time taken [Writing] = {((time.time() - start) / 60):.5f} seconds")
    print("Captions created !")
    print(f"Prepaid data:\n\tTraining: {len(data_to_write['train'])}\n\tValidation: {len(data_to_write['valid'])}")


def create_captions(dataSetPath, Metadata):

    if not os.path.exists(f"{dataSetPath}/metadata.csv") or \
            not os.path.exists(f"{dataSetPath}/metadata_train.csv") or \
            not os.path.exists(f"{dataSetPath}/metadata_valid.csv"):


        data_to_write = process_metadata(dataSetPath)

        ## Filtering when limit = True
        if Metadata.Limit:
            data_to_write = limit_on_total(Metadata, data_to_write)

        write_metadata(dataSetPath, data_to_write)

if __name__ == '__main__':

    with open('../config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

    create_captions(Args.Common.DataSet.Path, Args.Metadata)