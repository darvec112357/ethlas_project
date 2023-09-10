import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFRobertaForSequenceClassification
import pandas as pd
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

tokenizer = AutoTokenizer.from_pretrained("mariagrandury/roberta-base-finetuned-sms-spam-detection")
custom_objects = {"TFRobertaForSequenceClassification": TFRobertaForSequenceClassification}
model = tf.keras.models.load_model('model.h5', custom_objects=custom_objects)

def message_to_input(text):
    inputs = tokenizer(
        text,
        max_length=80,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
        return_tensors="tf",
        return_attention_mask=True,
        return_token_type_ids=False,
        verbose=True
    )
    return inputs

def predict(text):
    input_data = message_to_input(list(text))
    prediction = model.predict({'input_ids': input_data['input_ids'], 'attention_mask': input_data['attention_mask']})
    prediction = np.argmax(prediction,axis=1)
    return prediction

if __name__ == '__main__':
    path = sys.argv[1]
    test_size = int(sys.argv[2])
    test_df = pd.read_csv(path)
    x_test = test_df['text'][0:test_size]
    y_true = test_df['label'][0:test_size]
    if test_df.shape[1] > 2:
        raise Exception('Input Format Unexpected')
    test_predictions = pd.DataFrame(predict(x_test))
    test_predictions.to_csv('predictions.csv',index=False)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true,test_predictions))
    print("Classification Report:")
    print(classification_report(y_true,test_predictions))
