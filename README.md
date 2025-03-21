# 📰 NLP-Project: Real or Fake News Classifier

**Real or Fake news?** No worries — our **Natural Language Processing (NLP)** model will decide for you!  
We developed a robust **text classification model** capable of distinguishing between real and fake news articles, leveraging advanced NLP techniques.

[Task Descriptions and Project Instructions](https://github.com/ironhack-labs/project-nlp-challenge)

![Fake News Classifier](images/fakenewsimage.jpg)

---

##  About the Project

This project focuses on building a **machine learning pipeline for fake news detection**. Using NLP methods and classification algorithms, we trained and evaluated a model that can analyze news headlines and predict their authenticity with high accuracy.

---

##  Features

- **Text preprocessing**: tokenization, stopwords removal, stemming, lemmatization, and vectorization.
- **Model training**: algorithms such as **Support Vector Machine (SVM)**, **Logistic Regression**, **Naive Bayes**, and **Random Forest**.
- **Experimentation** with embedding-based (Word2Vec and GloVe) and TF-IDF-based models.
- **Evaluation metrics**: accuracy, precision, recall, F1-score, R² score.

![Model Comparison Chart](images/chartnpl.png)

---

##  Dataset

We worked with **two datasets provided by our professor**:
- **Labeled dataset** (`training_data_lowercase.csv`): contains real and fake news headlines for training and evaluation.
- **Unlabeled dataset** (`testing_data_lowercase_nolabels.csv`): used for generating predictions and testing the model in real-world scenarios.

---

## 📁 Repository Structure

- **SVM (winner model)** notebook and predictions.
- **Multinomial Naive Bayes** notebook and predictions.
- **Random Forest Classifier** notebook and predictions.
- **Logistic Regression** notebook and predictions.
- **Global Vectors for Word Representation (embeddings)** notebook and predictions.
- **Word2Vec embeddings** notebook and predictions.
- **Images** used in README.
- **Presentation** of the project.
- **Labeled dataset**: `training_data_lowercase.csv`.
- **Unlabeled dataset**: `testing_data_lowercase_nolabels.csv`.

---

## ⚙️ Installation

Use `requirements.txt` to install the required packages. We recommend using a virtual environment.

```bash
python -m venv .venv
.venv/Scripts/activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```
## 📚 NLTK Data Resources
Before running the notebooks, ensure you download necessary NLTK resources by running:

```bash
python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
## 📥GloVe Embeddings Download
If you want to use Global Vectors for Word Representation (GloVe) in this project, you need to manually download the GloVe pre-trained embeddings.

You can download them from the official Stanford NLP website:
👉 https://nlp.stanford.edu/projects/glove/

⚠️ Note:
After downloading, make sure to place the GloVe files (glove.6B.300d.txt) inside a folder named load_glove_embeddings/ in the root directory of this project or update the path in the corresponding notebook.


