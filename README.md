# 🎬 Rotten Tomatoes Movies: Audience Rating Classification 🎯  

## Project Overview  
This project focuses on building a **classification model** to predict the **audience rating** category of movies. Using the Rotten Tomatoes dataset, the goal is to classify movies into different rating levels based on features like critic rating, genre, runtime, and release year.  

---

## Table of Contents  
1. [About the Dataset](#about-the-dataset)  
2. [Objective](#objective)  
3. [Technologies Used](#technologies-used)  
4. [Project Workflow](#project-workflow)  
5. [Model Performance](#model-performance)  
6. [How to Run the Project](#how-to-run-the-project)  
7. [Conclusion](#conclusion)  

---

## About the Dataset 📊  
The **Rotten Tomatoes Movies Dataset** contains information about movies, including:  
- **Movie Title**  
- **Critic Rating** (Rotten Tomatoes)  
- **Audience Rating** (target variable as categories: e.g., High, Medium, Low)  
- **Genre**  
- **Release Year**  
- **Runtime**  
- Additional metadata  

**Dataset File**: `Rotten_Tomatoes_Movies3.xls`  

---

## Objective 🎯  
The primary objective of this project is:  
> To **classify movies** into categories (e.g., High, Medium, Low) based on the **audience rating** using machine learning models.  

---

## Technologies Used 🛠️  
The following tools and libraries were used in this project:  

- **Python** 🐍  
- **Pandas** for data manipulation  
- **NumPy** for numerical operations  
- **Scikit-Learn** for building and evaluating the classification model  
- **Matplotlib** and **Seaborn** for data visualization  
- **Jupyter Notebook** for code development and execution  

---

## Project Workflow 🔄  
The project follows these key steps:  

### 1. **Data Preprocessing**  
- Handling missing values in the dataset.  
- Encoding categorical variables like **genre** using One-Hot Encoding or Label Encoding.  
- Splitting the dataset into **training** and **testing** sets.  

### 2. **Exploratory Data Analysis (EDA)**  
- Analyzing the distribution of audience rating categories.  
- Identifying relationships between **critic ratings**, **runtime**, and audience ratings.  
- Visualizing correlations using bar plots, scatter plots, and heatmaps.  

### 3. **Model Preparation**  
- Building classification models:  
  - Logistic Regression  
  - Decision Tree Classifier  
  - Random Forest Classifier  
- Splitting the dataset into training and testing sets (e.g., 80%-20%).  

### 4. **Model Evaluation**  
- Evaluating model performance using:  
  - **Accuracy Score**  
  - **Confusion Matrix**  
  - **Classification Report** (Precision, Recall, F1-Score)  

---

## Model Performance 📈  
The model achieved the following performance:  

- **Model Used**: Random Forest Classifier  
- **Accuracy Score**: `90%`   


## How to Run the Project 🚀  
Follow these steps to run the project on your system:  

1. **Clone the repository**  
   ```bash
   git clone https://https://github.com/Sruteka/Rotten_Tomatoes_Movies3.git
   cd Rotten_Tomatoes_Movies3

2. **Install required libraries**
Use pip to install the necessary dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn

3. **Run the Jupyter Notebook**
Open the notebook to view the full analysis and predictions:

   ```bash
   jupyter notebook

4. **Dataset File:**
Ensure the file Rotten_Tomatoes_Movies3.xls is in the same directory as your notebook.

Conclusion 📝
This project successfully predicts the audience rating category of movies using classification techniques. The model provides reliable accuracy and insights into movie features such as critic ratings, genres, and runtime, helping to identify patterns in audience preferences.

Author 💻
Sruteka PJ
Data Science Enthusiast

LinkedIn: www.linkedin.com/in/sruteka-pj-a50a14266
GitHub: https://github.com/Sruteka

License 📜
This project is licensed under the MIT License. Feel free to use and modify it for your own purposes.


---

### **Key Changes**
1. **Classification Focus**: Updated the README to reflect a **classification** project instead of regression.  
2. **Accuracy Score**: Highlighted accuracy as the main evaluation metric.  
3. **Confusion Matrix**: Added mentions of other evaluation techniques like confusion matrix and classification report.  

---

Would you like help adding **code snippets** for model training or evaluating the confusion matrix? Let me know! 😊

