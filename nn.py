import argparse
import datasets
import pandas as pd
import transformers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Model
from transformers import AutoTokenizer, TFBertForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np

# Load the tokenizer from DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(examples):
    """Converts the text of each example to a sequence of integers
    representing token ids."""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")

def train(model_path="model", train_path="train.csv", dev_path="dev.csv"):
    # Load the CSVs into Huggingface datasets to allow use of the tokenizer
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # The labels are the names of all columns except the first
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        return {"labels": [float(example[l]) for l in labels]}

    # Convert text and labels to format expected by model
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Convert Huggingface datasets to Tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': hf_dataset['train']['input_ids'],
            'attention_mask': hf_dataset['train']['attention_mask']
        },
        hf_dataset['train']['labels']
    )).batch(32)

    dev_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': hf_dataset['validation']['input_ids'],
            'attention_mask': hf_dataset['validation']['attention_mask']
        },
        hf_dataset['validation']['labels']
    )).batch(32)

    # Define the model architecture
    input_ids = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name='attention_mask')

    bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    bert_output = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state

    lstm_layer = Bidirectional(LSTM(64, dropout=0.2))(bert_output)
    output_layer = Dense(len(labels), activation='sigmoid')(lstm_layer)
    
    model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output_layer)

    # Compile the model with hyperparameters
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metric = tf.keras.metrics.F1Score(average="micro", threshold=0.5)

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Fit the model to the training data, monitoring performance on the dev data
    model.fit(train_dataset, epochs=2, validation_data=dev_dataset, callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_f1_score",
                mode="max",
                save_best_only=True)])

    # Save the model
    model.save(model_path)

def predict(model_path="model", input_path="test-in.csv"):
    
    # Custom object needed for loading the model
    custom_objects = {'TFDistilBertModel': TFDistilBertModel}

    # Load the saved model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)


    # Load the data for prediction
    df = pd.read_csv(input_path)

    # Create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    tf_dataset = tf.data.Dataset.from_tensor_slices({
        'input_ids': hf_dataset['input_ids'],
        'attention_mask': hf_dataset['attention_mask']
    }).batch(16)

    # Generate predictions from model
    predictions = model.predict(tf_dataset)
    predictions = np.where(predictions > 0.5, 1, 0)

    # Assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # Write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name='submission.csv'))
    
if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()  
