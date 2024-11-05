from process_datasets import build_dataset, build_metrics, collate_fn, collate_ImageNet_fine_tuning_fn, collate_coco_fn_tuning_fn
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, Pix2StructProcessor, logging
from models_utils import ViTForImageClassification, DeiTForImageClassification, Pix2StructForConditionalGeneration
from transformers.training_args import OptimizerNames
from utils.pathUtils import prepare_output_path, get_checkpoint_path
from utils.commonUtils import start_training
import warnings

warnings.filterwarnings('ignore')


def get_fine_tuning_trainer_args(output_path, hyperparameters, args_fn):

    return args_fn(
        output_dir=output_path + 'training/',
        logging_dir=output_path + 'logs/',
        per_device_train_batch_size=hyperparameters.TrainBatchSize,
        per_device_eval_batch_size=hyperparameters.EvalBatchSize,
        evaluation_strategy="steps",
        num_train_epochs=hyperparameters.Epochs,
        save_steps=hyperparameters.Steps.SaveSteps,
        eval_steps=hyperparameters.Steps.EvalSteps,
        logging_steps=hyperparameters.Steps.LoggingSteps,
        learning_rate=hyperparameters.Lr,
        lr_scheduler_type='cosine',
        warmup_ratio=hyperparameters.WarmUpRatio,
        weight_decay=hyperparameters.WeightDecay,
        save_total_limit=2,
        metric_for_best_model='bleu',
        greater_is_better=True,
        optim=OptimizerNames.ADAMW_HF,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        seed=42,
        half_precision_backend="auto",
        gradient_accumulation_steps=hyperparameters.Steps.GradientAccumulation,
        predict_with_generate=True
    )


def fine_tuning(Args):
    num_labels, training_data, testing_data = build_dataset(True, Args)

    output_path = prepare_output_path('FineTuned', Args)

    model = Args.FineTuning.Model.Name

    if "pix" in model:
        classificationMode = Pix2StructForConditionalGeneration
    else:
        if "deit" in model and "distilled" in model:
            classificationMode = DeiTForImageClassification
        else:
            classificationMode = ViTForImageClassification

    if Args.FineTuning.Model.LoadCheckPoint:
        model = get_checkpoint_path('FineTuned', Args)

    if "pix" in model:
        pretrained_model = classificationMode.from_pretrained(model, cache_dir=Args.FineTuning.Model.CachePath)
    else:
        pretrained_model = classificationMode.from_pretrained(model, num_labels=num_labels,
                                                              cache_dir=Args.FineTuning.Model.CachePath,
                                                              ignore_mismatched_sizes=True)

    compute_metrics = build_metrics(Args.Common.Metrics, Args.FineTuning.Model)

    if Args.Common.DataSet.Name == "coco":

        fine_tune_args = get_fine_tuning_trainer_args(output_path, Args.FineTuning.Hyperparameters,
                                                      Seq2SeqTrainingArguments)

        fine_tune_trainer = Seq2SeqTrainer(
            model=pretrained_model,
            args=fine_tune_args,
            compute_metrics=compute_metrics,
            data_collator=collate_coco_fn_tuning_fn,
            train_dataset=training_data,
            eval_dataset=testing_data,
            processing_class=Pix2StructProcessor.from_pretrained(model, cache_dir=Args.FineTuning.Model.CachePath)
        )
    else:

        fine_tune_args = get_fine_tuning_trainer_args(output_path, Args.FineTuning.Hyperparameters,
                                                      TrainingArguments)

        fine_tune_trainer = Trainer(
            model=pretrained_model,
            args=fine_tune_args,
            compute_metrics=compute_metrics,
            data_collator=collate_ImageNet_fine_tuning_fn if Args.Common.DataSet.Name == "imageNet" else collate_fn,
            train_dataset=training_data,
            eval_dataset=testing_data,
        )

    start_training(Args, fine_tune_trainer, Args.FineTuning.Model.LoadCheckPoint, model, output_path,
                   Args.FineTuning.Model.OutputPath, testing_data)
