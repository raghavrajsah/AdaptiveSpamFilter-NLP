# Adaptive Spam Filter: NLP Final Project

## Project Overview
This repository contains code and resources for our NLP final project: an adaptive email/SMS spam filtering system. Our goal is to create a filter that not only detects spam but also categorizes it by severity, supports multiple languages, and adapts based on user feedback. We utilize NLP techniques and machine learning models to provide a robust and explainable spam detection solution.

## Repository Structure
- **`spam sample data.tsv`**  
  Sample TSV data file required for training and testing in NLPScholar.
- **`text_classification.yaml`**  
  Configuration file containing parameters for text classification with NLPScholar.
- **`train.pbs`**  
  Job submission file for the training phase on Turing.
- **`evaluate.pbs`**  
  Job submission file for the evaluation phase on Turing.
- **`README.md`**  
  Main README file, outlining the project and setup instructions.

## Getting Started
### Prerequisites
- **NLPScholar Toolkit**  
  Ensure to clone NLPScholar. Follow [this link](https://github.com/forrestdavis/NLPScholar/tree/main) to set it up if needed.
- **Python 3.8+**  
  For data preparation and evaluation scripts.
