# from flask import Flask, render_template, request
# import torch
# from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
# import nltk
# import re
# import tensorflow as tf

# app = Flask(__name__)

# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512)

# # Load your trained language model here
# # Replace 'load_model' with the actual code to load your model

# model = torch.load('model/model.pt', map_location=torch.device('cpu'))
# model.eval()  # Set the model to evaluation mode
    

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     message = request.form['message']

#     # Tokenize and preprocess the message
#     inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
    
#     # Make a prediction using the model
#     with torch.no_grad():
#         logits = model(**inputs).logits
#         probabilities = logits.softmax(dim=-1)
#         prediction = "Phishing" if probabilities[0][1] > 0.5 else "Legitimate"

#     return render_template('result.html', prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
from transformers import TFRobertaForSequenceClassification


app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("mariagrandury/roberta-base-finetuned-sms-spam-detection")
custom_objects = {"TFRobertaForSequenceClassification": TFRobertaForSequenceClassification}
model = tf.keras.models.load_model('model/model.h5', custom_objects=custom_objects)


# Convert the processed message to a format suitable for model input
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    input_data = message_to_input(message)

    prediction = model.predict({'input_ids': input_data['input_ids'], 'attention_mask': input_data['attention_mask']})
    prediction = np.argmax(prediction,axis=1)
    final_prediction = 'Phishing' if prediction == 0 else 'Ham'

    return render_template('result.html', prediction=final_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

