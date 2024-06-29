from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import pipeline
from datacollection import resume
from sklearn.metrics import classification_report
import numpy as np

tokenizer  = AutoTokenizer.from_pretained("bert-base-uncased")

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

pipeline = pipeline("ner", model=model,tokenizer=tokenizer )


def ner(text):
    return pipeline(text)


entities = [ner(w) for w in resume ]

training_args = TrainingArguments(
     output_dir='./results',  
     num_train_epochs=3,        
     per_device_train_batch_size=16, 
     per_device_eval_batch_size=64,
     warmup_steps=500,
     weight_decay=0.01, 
     logging_dir='./logs',
     logging_steps=10,
)

trainer =  Trainer(
    model=model,
    args= training_args,
    train_dataset= train_dataset,
    eval_dataset= eval_dataset
)

trainer.train()

def evaluate(model, eva_dataset):
    predictions, labels, _= trainer.predict(eva_dataset)
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    print(classification_report(true_labels, true_predictions))

evaluate(model, eval_dataset)