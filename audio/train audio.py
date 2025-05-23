
import logging
import numpy as np
import torch
import os
from datasets import load_from_disk
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Trainer,
    TrainingArguments,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_CHECKPOINT = "facebook/wav2Vec2-large-960h"
OUTPUT_DIR = "./wav2vec2-cv17-uk-finetuned-standard"
LOGGING_DIR = "./logs_cv17_uk_standard"
SAVED_MODEL_DIR = "model_cv17_uk_final_standard"
PROCESSED_DATASET_PATH = "/Users/mac/PycharmProjects/FallDetect/audio/processed_cv17_uk_dataset"

try:
    NUM_PROC_LOADER = max(1, os.cpu_count() // 4 if os.cpu_count() else 1)
except AttributeError:
    NUM_PROC_LOADER = 1

TRAIN_BATCH_SIZE_PER_DEVICE = 4
EVAL_BATCH_SIZE_PER_DEVICE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
NUM_TRAIN_EPOCHS = 40
EVAL_SAVE_LOG_STEPS = 500
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 2
try:

    metrics_loaded = True
except Exception as e:
    logging.warning(f"Could not load WER/CER metrics (pip install evaluate jiwer): {e}. Evaluation will only show loss.")
    metrics_loaded = False

def compute_metrics(pred):
    if 'processor' not in globals():
        logging.error("Processor not found globally for compute_metrics!")
        return {"eval_loss": pred.metrics.get("eval_loss", -1)}

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)


if __name__ == "__main__":

    logging.info("Script execution started inside __main__ block.")
    logging.info(f"Attempting to load processed dataset from disk: {PROCESSED_DATASET_PATH}")
    if not os.path.exists(PROCESSED_DATASET_PATH):
        logging.error(f"Processed dataset not found: {PROCESSED_DATASET_PATH}")
        exit(1)
    try:
        dataset = load_from_disk(PROCESSED_DATASET_PATH)
        logging.info("Processed dataset loaded successfully from disk.")
        required_processed_cols = ["input_values", "labels", "length"]
        if "train" not in dataset or not all(col in dataset["train"].column_names for col in required_processed_cols):
             logging.error(f"Loaded dataset is missing required columns: {required_processed_cols}.")
             exit(1)
        else:
             logging.info(f"Required columns {required_processed_cols} found.")
             logging.info(f"Loaded dataset structure:\n{dataset}")
    except Exception as e:
        logging.error(f"Failed to load dataset from disk: {e}", exc_info=True)
        exit(1)

    logging.info(f"Loading processor from checkpoint: {MODEL_CHECKPOINT}")
    try:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_CHECKPOINT)
    except Exception as e:
        logging.error(f"Failed to load processor: {e}", exc_info=True)
        exit(1)

    logging.info(f"Loading model from checkpoint: {MODEL_CHECKPOINT}")
    try:
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_CHECKPOINT)
        processor_vocab_size = len(processor.tokenizer)
        model_vocab_size = model.config.vocab_size
        if model_vocab_size != processor_vocab_size:
            logging.warning(f"Resizing model output layer 'lm_head' from {model_vocab_size} to {processor_vocab_size}.")
            model.resize_token_embeddings(processor_vocab_size)
            model.config.vocab_size = processor_vocab_size
        else:
            logging.info(f"Processor vocab size ({processor_vocab_size}) matches model vocab size.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        exit(1)

    # === КРОК 3: Data Collator (Використовуємо стандартний DataCollatorForCTC) ===
    logging.info("Initializing Data Collator (using standard DataCollatorForCTC)...")
    try:

        logging.info("DataCollatorForCTC initialized successfully.")
    except NameError: # На випадок, якщо імпорт не спрацював через СЕРЙОЗНІ проблеми середовища
         logging.error("!!! FAILED to use DataCollatorForCTC despite expecting it in this transformers version !!!")
         logging.error("Please check your 'transformers' installation and environment thoroughly.")
         exit(1)
    except Exception as e:
         logging.error(f"Error initializing DataCollatorForCTC: {e}", exc_info=True)
         exit(1)


    logging.info("Defining Training Arguments (modern API)...")
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()
    fp16_enabled = use_cuda

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        # Сучасні стратегії
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=EVAL_SAVE_LOG_STEPS,
        save_steps=EVAL_SAVE_LOG_STEPS,
        logging_steps=max(10, EVAL_SAVE_LOG_STEPS // 10),
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=fp16_enabled,
        group_by_length=True,
        dataloader_num_workers=NUM_PROC_LOADER,
        # Відстеження найкращої моделі
        load_best_model_at_end=True,
        metric_for_best_model="wer" if metrics_loaded else "eval_loss",
        greater_is_better=False,
        # Інше
        logging_dir=LOGGING_DIR,
        report_to="tensorboard",
        resume_from_checkpoint=False,
        do_eval=True
    )
    logging.info(f"Training Arguments defined. Using dataloader_num_workers={NUM_PROC_LOADER}")

    logging.info("Initializing Trainer...")
    eval_dataset = None
    eval_dataset_key = None
    if "validation" in dataset and len(dataset["validation"]) > 0:
        eval_dataset_key = "validation"
    elif "test" in dataset and len(dataset["test"]) > 0:
        eval_dataset_key = "test"

    if eval_dataset_key:
        eval_dataset = dataset[eval_dataset_key]
        logging.info(f"Using '{eval_dataset_key}' split for evaluation.")
        training_args.do_eval = True
    else:
        logging.warning("No suitable evaluation dataset found ('validation' or 'test'). Disabling evaluation and load_best_model_at_end.")
        training_args.do_eval = False
        training_args.load_best_model_at_end = False # Немає сенсу без оцінки

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer, # Для збереження
        compute_metrics=compute_metrics if metrics_loaded else None, # Передаємо функцію метрик
    )
    logging.info("Trainer initialized.")

    # === КРОК 6: Тренування ===
    logging.info("Starting training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logging.info("Training finished successfully.")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        try:
            logging.warning("Attempting to save state after training error...")
            error_save_path = os.path.join(OUTPUT_DIR, "state_on_error")
            trainer.save_model(error_save_path)
            processor.save_pretrained(error_save_path)
            logging.info(f"Model/processor state saved to {error_save_path}")
        except Exception as save_e:
            logging.error(f"Could not save state after error: {save_e}")
        exit(1)

    if training_args.do_eval and eval_dataset is not None:
        logging.info("Evaluating final model (best checkpoint)...")
        try:
            eval_metrics = trainer.evaluate()
            logging.info(f"Evaluation results: {eval_metrics}")
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        except Exception as e:
            logging.error(f"Error during final evaluation: {e}", exc_info=True)
    else:
        logging.info("Skipping final evaluation.")

    logging.info(f"Saving final model (best checkpoint if load_best_model_at_end=True) and processor to {SAVED_MODEL_DIR}...")
    try:
        trainer.save_model(SAVED_MODEL_DIR)
        processor.save_pretrained(SAVED_MODEL_DIR)
        logging.info(f"Final model and processor saved successfully to {SAVED_MODEL_DIR}")
    except Exception as e:
        logging.error(f"Error saving final model: {e}", exc_info=True)

    logging.info("Script finished.")
