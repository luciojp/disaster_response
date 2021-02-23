# 🚧  Disaster Response Pipeline Project 🚧  

### Table of Contents

1. [About project](#about)
2. [Installation](#installation)
3. [Running](#running)
4. [Repository Structure](#repo)
6. [Model](#model)
7. [Results](#results)


## 💻 About project <a name="about"></a>

When a disaster occurs the organizations are Bombarded with many messages and
at this time is really hard to them to classify what are the most important messages
that usually is only one in every thousand messages.
So to try to help the organizations at this task I built a supervised machine learning
model using the [Figure Eight Inc.](https://www.figure-eight.com/) pre labeled tweets and text messages data.


## 🛠 Installation <a name="installation"></a>

Create a virtual environment named **disaster_env**.

```
$ python3 -m env disaster_env -- for Linux and macOS
$ python -m env disaster_env -- for Windows
```

After that, activate the python virtual environment

```
$ source disaster_env/bin/activate -- for Linux and macOS
$ disaster_env\Scripts\activate -- for Windows
```

Install the requirements

```
$ pip install -r requirements.txt
```

## 🎲 Running <a name="running"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:8080/

## Repository Structure <a name="repo"></a>

- The `data` folder contains the disaster's data and the script to clean and store the data.
- The `models` have the script to train and classify the data.
- The `app` folder with all the python scripts and html to build and run the web project.
- The `requirements.txt` has the needed packages to run the notebook.

## Model <a name="model"></a>

It was used Pipeline and FeatureUnion to build a XGBClassifier as you can see bellow.

```
pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('num_verbs', extractor_transformers.CountPosTagTransformer('VB')),
            ('num_nouns', extractor_transformers.CountPosTagTransformer('JJ')),
            ('num_adjectives',
             extractor_transformers.CountPosTagTransformer('PRP')),
            ('num_pronouns',
             extractor_transformers.CountPosTagTransformer('NN'))
        ])),

        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])
```

The `CountPosTagTransformer` counts the number of words of a specific type at the text.

It is also possible to realize a Grid Search at the code, adding the desired parameters at line 107 of train_classifier.py.

 ```
 parameters = {
        'clf__estimator__learning_rate': [0.1, 0.01],
    }
 ```

 Note, that if you add a lot of parameters you have to have a robust setup to run the code.

## 📝 Results <a name="results"></a>

The result found using XGBoost and the features looks for above:

| Model   | weighted-average F1-score | 
| --------| --------------------------| 
| XGBoost | 0.938832 |
