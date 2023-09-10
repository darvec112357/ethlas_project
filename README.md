This task is not easy, because almost every aspect of it is new to me, from LLM to flask, docker (I did have experience with docker during an internship three years ago, but barely remember it). 

# Phishing Message Case-Study
## 1. Install python requirements

```
pip install -r requirements.txt
```

## 2. Dataset
[Phishing Email](https://www.kaggle.com/datasets/subhajournal/phishingemails) was chosen as the dataset for training. It consists of 18650 rows, with a 3-7 split between normal email and phishing email. Do note the difference between spam and phishing, since there are lots of datasets of spam email, but not so many about phishing.

Run the following command to preprocess the data, which removes duplicates, empty values and changes the labels in terms of binary representation. In the end, a train.csv will be created.
```
python preprocess.py
```

## 3. Training

The training was run on a Kaggle kernel with GPU T4*2, which took roughly an hour to finish. The model used is based on Roberta. After training, a model.h5 will be created. Copy this model into UI/model for prediction purpose.
```
python train.py
```

## 4. Evaluation

Run the following command to evaluate the performance of the trained model. Where **data.csv** is the data you would like to evaluate on (there should be only two columns, namely **text** and **label**) and **test_size** is the number of rows which you want to predict. For large datasets, you can keep this size small enough so that the prediction does not take too long.
```
python predict.py data.csv test_size
```

## 5. Integrate into Flask.
Run the following command will create a running server. Users can enter any text they want to check and the model will predict an output, either being Ham or Phishing.
```
python app.py
```

## 6. Deploy on Docker
```
cd UI
```
Run the following command to build a docker image.
```
docker image build -t flask_docker .
```
Finally we can run the docker image, which will create the expected server and UI functionalities.
```
docker run -p 3000:3000 -d flask_docker
```