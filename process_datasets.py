import evaluate
import torch
import json
import yaml
import requests
from datasets import load_dataset
from utils.argUtils import CustomObject, get_yaml_loader
from transformers import ViTImageProcessor, DeiTImageProcessor, Pix2StructProcessor
from PIL import Image
from io import BytesIO


with open('config.yaml', 'r') as file:
    config = yaml.load(file, get_yaml_loader())

Args = json.loads(json.dumps(config), object_hook=lambda d: CustomObject(**d))


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


def collate_imageNet_fn(batch):
    return {
        'inputPath': [x['inputPath'] for x in batch],
        'labels': torch.tensor([x['label'] for x in batch])
    }


def collate_ImageNet_fine_tuning_fn(batch):

    collated_inputs = collate_imageNet_fn(batch)

    return processInputs(collated_inputs, Args.FineTuning.Model)


def processInputs(inputs, Model):

    if 'deit' in Model.Name:
        feature_extractor = DeiTImageProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)
    else:
        feature_extractor = ViTImageProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)

    images = [Image.open(Args.Common.DataSet.Path + '/' + path) for path in inputs['inputPath']]
    batches = [img.convert("RGB") if img.mode != 'RGB' else img for img in images]
    image_inputs = feature_extractor(batches, return_tensors='pt')
    return {
        'pixel_values': image_inputs['pixel_values'],
        'labels': inputs['labels']
    }


def collate_coco_fn(batch):
    return {
        'inputPath': [x['inputPath'] for x in batch],
        'labels': [x['label'] for x in batch]
    }


def collate_coco_fn_tuning_fn(batch):

    collated_inputs = collate_coco_fn(batch)

    return processCocoInputs(collated_inputs, Args.FineTuning.Model)


def processCocoInputs(inputs, Model):

    feature_extractor = Pix2StructProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)

    images = [Image.open(BytesIO(requests.get(url).content)) for url in inputs['inputPath']]
    batches = [img.convert("RGB") if img.mode != 'RGB' else img for img in images]
    image_inputs = feature_extractor(images=batches, return_tensors='pt')

    image_inputs['labels'] = feature_extractor(text=inputs['labels'], return_tensors="pt", padding=True).input_ids

    return image_inputs


def build_metrics(metric_args, Model=None):
    metrics_to_evaluate = metric_args.Name.split(',')
    for m in metrics_to_evaluate:
        _ = evaluate.load('custom_metrics/' + m, cache_dir=metric_args.CachePath, trust_remote_code=True)

    # accuracy = evaluate.load("accuracy", cache_dir='metrics/', trust_remote_code=True)

    metric = evaluate.combine(['custom_metrics/' + m for m in metrics_to_evaluate])

    if Model is not None:
        feature_extractor = Pix2StructProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)

    def compute_metrics(p):

        if feature_extractor:
            predictions = feature_extractor.batch_decode(p.predictions, skip_special_tokens=True)
            references = feature_extractor.batch_decode(p.label_ids, skip_special_tokens=True)
        else:
            predictions = p.predictions
            references = p.label_ids

        return metric.compute(
            predictions=predictions,
            references=references,
            # labels=list(range(p.predictions.shape[1])),
        )

    return compute_metrics


def build_dataset(is_train, Args, show_details=True):
    DataSet = Args.Common.DataSet
    if Args.FineTuning.Action:
        Model = Args.FineTuning.Model
    elif Args.Distillation.Action:
        Model = Args.Distillation.Model
    else:
        Model = Args.Visualization.Model

    label_key = DataSet.Label

    if DataSet.Name in ['imageNet','coco']:
        def preprocess(batchImage):
            pass
    else:
        if 'deit' in Model.Name:
            feature_extractor = DeiTImageProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)
        else:
            feature_extractor = ViTImageProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)

        def preprocess(batchImage):
            inputs = feature_extractor(batchImage['img'], return_tensors='pt')
            inputs['label'] = batchImage[label_key]
            return inputs

    prepared_train = None

    if is_train:

        if DataSet.Name in ['imageNet','coco']:
            dataset_train = load_dataset('csv', split=f"train[:{DataSet.Train}]", verification_mode='no_checks',
                                         data_files={"train":DataSet.Path + "/metadata_train.csv"})
            prepared_train = dataset_train
        else:
            dataset_train = load_dataset(DataSet.Name, split=f"train[:{DataSet.Train}]", verification_mode='no_checks',
                                         cache_dir=DataSet.Path + "/train")
            prepared_train = dataset_train.with_transform(preprocess)

        num_training_labels = len(set(dataset_train[label_key]))

        if show_details:
            print(f"\nTraining info:{dataset_train}")
            # print(f"\tNumber of labels = {num_training_labels}, {dataset_train.features[label_key]}")

    if DataSet.Name in ['imageNet','coco']:
        dataset_test = load_dataset('csv', split=f"validation[:{DataSet.Test}]",
                                    data_files={"validation":DataSet.Path + "/metadata_valid.csv"})

        prepared_test = dataset_test
    else:
        dataset_test = load_dataset(DataSet.Name, split=f"test[:{DataSet.Test}]", verification_mode='no_checks',
                                    cache_dir=DataSet.Path + "/test")

        prepared_test = dataset_test.with_transform(preprocess)

    num_validation_labels = len(set(prepared_test[label_key]))
    if show_details:
        print(f"\nTesting info:{dataset_test}")
        # print(f"\tNumber of labels = {num_validation_labels}, {dataset_test.features[label_key]}")

    if is_train:
        return num_training_labels, prepared_train, prepared_test
    else:
        return num_validation_labels, None, prepared_test
