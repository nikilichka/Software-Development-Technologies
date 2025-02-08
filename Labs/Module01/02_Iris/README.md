# Collaborative Machine Learning Project with Git  

## Overview  
This project is designed as a collaborative machine learning exercise using Git. Students will work in pairs to develop a machine learning pipeline with clearly defined roles and responsibilities.  

## Objectives  
- Practice Git-based collaboration by working on feature branches.  
- Implement a machine learning pipeline, including data preprocessing, model training, and evaluation.  
- Learn and follow best practices for Git workflows.  

---

## Project Structure  
The repository includes the following files:  
- `data_preprocessing.py` - Script for data preprocessing (handled by Student A).  
- `model_training.py` - Script for model training (handled by Student B).  
- `model_evaluation.py` - Script for evaluating the trained model (handled by Student B).  
- `.gitignore` - Ensures large files (e.g., datasets, models) and temporary files are not pushed to the repository.  
- `README.md` - Documentation for the project.  

---

## Roles and Responsibilities  
- **Student A: Data Preprocessing**  
  - Write and test the `data_preprocessing.py` script.  
  - Ensure the dataset is preprocessed and saved locally as `cleaned_data.csv` (do not push this file).  

- **Student B: Model Training and Evaluation**  
  - Write and test the `model_training.py` and `model_evaluation.py` scripts.  
  - Train the model using the preprocessed data and save the trained model as `trained_model.pkl`.  
  - Evaluate the model and generate a confusion matrix image.  

---

## Git Workflow  
1. **Clone the Repository**  
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   git pull origin main
