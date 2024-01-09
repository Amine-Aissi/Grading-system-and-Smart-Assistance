<img src="https://github.com/Amine-Aissi/Grading-system-and-Smart-Assistance/blob/main/images/banner.png" alt="Your Image" width="100%">


# Grading-system-and-Smart-Assistance
Arabic Automated short answers grading system and Smart Assistance for islamic education for schoolers.

## Table of contents

[Scraping Data](#scraping-data)

[Establishment of Arabic Natural language processing pipeline](#establishment-of-arabic-natural-language-processing-pipeline)

[Exploratory data analysis (EDA)](#exploratory-data-analysis)
* [Loading Data](#loading-data)
* [Characteristics Of Dataset](#characteristics-of-dataset)
* [Data Cleaning](#data-cleaning)

[Data Pre-Preprocessing](#data-pre-preprocessing)
* [Split Data](#split-data)

[build Models](#build-models)
* [Testing  Models](#testing-models)
* [Saving The Best Model](#saving-the-best-model)

[Backend of our Application](#backend-of-our-application)
* [FastAPI](#fastapi)
* [GraphQL Server Side](#graphql-server-side)
* [Docker Server Side](#docker-server-side)

[Frontend of our Application](#frontend-of-our-application)
* [Angular](#angular)
* [GraphQL Client Side](#graphql-client-side)
* [Docker Client Side](#docker-client-side)

[Testing the Application](#testing-the-application)
* [Home Page](#home-page)
* [Question Page](#question-page)
* [Result Page](#result-page)
[Contributors](#contributors)

# Arabic Automated Short Answers Grading System and Smart Assistance for Islamic Education

Arabic Automated short answers grading system and Smart Assistance for islamic
education for schoolers, the idea behind this system is to give the adequate grade to the students
according to their answers (grades from 0 to 20), the system should be in Arabic, and you should
prepare your own dataset by collecting row data from several resource by using the scraping
technics, the system should also assist the student by generating certain guidelines during the
process of evaluation according to their progression.

## Project Components

- **Scraping Data:**
  - Collects data from several sources, including Arabic websites, datasets, and books.

- **Natural Language Processing Pipeline:**
  - Establishes an Arabic natural language processing pipeline.

- **Exploratory Data Analysis (EDA):**
  - Applies EDA techniques to understand the collected data.

- **Word Embedding and Encoding:**
  - Implements word embedding and encoding.

- **Model Training:**
  - Utilizes RNN, LSTM, and Transformer models.

- **Text Generation for Student Assistance:**
  - Uses BERT (Fine Tuning), Prompt/Fine-tuning of LLMs like GPT 3, Falcon, and Alpaca.

- **Model Evaluation:**
  - Evaluates models using various metrics such as ROC, accuracy, f1 score, blue score, etc.

- **Model Deployment:**
  - Deploys the best model using Docker containers and Kubernetes orchestration.

- **Consumption via SAP Web Application (Angular):**
  - Provides an interface for users to interact with the system.

## Specific Tools Used


- NLTK
- Word2vec
- Glove
- And other tools as required for specific tasks.




## Scraping data

The initial step in our project involves collecting raw data from diverse sources, including Arabic websites, datasets, and educational books. This varied dataset is essential for training and evaluating our Arabic Automated Short Answers Grading System and Smart Assistance for Islamic education.

### Procedure

To gather relevant and comprehensive data, we employ scraping techniques, accessing Arabic websites, exploring existing datasets, and referencing educational books. This process ensures the creation of a dataset that captures the linguistic and educational nuances of the Arabic language.
