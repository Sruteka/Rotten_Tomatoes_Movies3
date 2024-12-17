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


Here’s a README for your AB_NYC_2019 project where you applied supervised and unsupervised learning techniques.

README for AB_NYC_2019 Project
markdown
Copy code
# 🏠 NYC Airbnb Data Analysis: Supervised & Unsupervised Learning 🧠  

## Project Overview  
This project analyzes the **AB_NYC_2019** dataset, which contains Airbnb listings in New York City. By applying **supervised** and **unsupervised learning**, we:  
1. **Predicted prices** of Airbnb listings (Supervised Learning).  
2. **Clustered listings** based on key features (Unsupervised Learning).  

The project provides insights into pricing patterns, popular neighborhoods, and groups Airbnb properties into meaningful clusters for analysis.

---

## Table of Contents  
1. [About the Dataset](#about-the-dataset)  
2. [Objective](#objective)  
3. [Technologies Used](#technologies-used)  
4. [Project Workflow](#project-workflow)  
5. [Supervised Learning: Price Prediction](#supervised-learning-price-prediction)  
6. [Unsupervised Learning: Clustering](#unsupervised-learning-clustering)  
7. [Conclusion](#conclusion)  
8. [How to Run the Project](#how-to-run-the-project)  
9. [Author](#author)  

---

## About the Dataset 📊  
The **AB_NYC_2019** dataset includes details about Airbnb listings in New York City, such as:  
- **ID** and **Name** of the listing  
- **Host ID** and **Host Name**  
- **Neighborhood Group** (e.g., Manhattan, Brooklyn)  
- **Neighborhood**  
- **Latitude** and **Longitude**  
- **Room Type** (e.g., Entire home/apt, Private room)  
- **Price** (Target variable for Supervised Learning)  
- **Minimum Nights**  
- **Number of Reviews**  

**Dataset File**: `AB_NYC_2019.csv`  

---

## Objective 🎯  
The project has two main objectives:  
1. **Supervised Learning**: Predict the **price** of Airbnb listings using machine learning models.  
2. **Unsupervised Learning**: Group Airbnb listings into meaningful **clusters** based on key features.  

---

## Technologies Used 🛠️  
The following tools and libraries were used:  

- **Python** 🐍  
- **Pandas** and **NumPy** for data manipulation  
- **Matplotlib** and **Seaborn** for data visualization  
- **Scikit-Learn** for machine learning models  
- **KMeans Clustering** for unsupervised learning  
- **Jupyter Notebook** for code development and execution  

---

## Project Workflow 🔄  
The project follows these key steps:  

### 1. **Data Preprocessing**  
- Handling missing values in features like `price` and `reviews`.  
- Encoding categorical variables like `neighbourhood_group` and `room_type` using One-Hot Encoding.  
- Feature scaling and transformation for numerical data.  

### 2. **Exploratory Data Analysis (EDA)**  
- Visualizing distributions of price, room types, and neighborhoods.  
- Identifying correlations between features using heatmaps.  

### 3. **Supervised Learning: Price Prediction**  
- **Target Variable**: `price`  
- **Features**: Room type, neighborhood, minimum nights, and reviews.  
- Applied models:  
   - **Linear Regression**  
   - **Random Forest Regressor**  
- **Model Evaluation**:  
   - R² Score  
   - Mean Absolute Error (MAE)  

### 4. **Unsupervised Learning: Clustering**  
- Used **KMeans Clustering** to group Airbnb listings based on:  
   - Latitude and Longitude (geographical features)  
   - Room Type  
   - Minimum Nights  
- Determined the optimal number of clusters using the **Elbow Method**.  
- Visualized clusters on a geographical map of New York City.  

---

## Supervised Learning: Price Prediction 📈  
The best-performing model achieved the following results:  

- **Model Used**: Random Forest Regressor  
- **R² Score**: `0.78`  
- **MAE**: `45.2`  

---

## Unsupervised Learning: Clustering 🗺️  
- Used **KMeans Clustering** with 4 clusters (optimal from Elbow Method).  
- Visualized clusters on NYC maps to identify patterns in property locations.  
- Insights:  
   - Cluster 1: Luxury listings in Manhattan.  
   - Cluster 2: Affordable listings in Brooklyn.  
   - Cluster 3: High-density private rooms in Queens.  

---

## Conclusion 📝  
The project successfully analyzed Airbnb data to:  
1. **Predict listing prices** using machine learning models.  
2. **Cluster properties** into meaningful groups for analysis.  

These insights help property owners, travelers, and Airbnb to understand price drivers and optimize their listings.

---

## How to Run the Project 🚀  
Follow these steps to run the project:  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/AB_NYC_2019_Price_Clustering.git
   cd AB_NYC_2019_Price_Clustering
Install Required Libraries
Use pip to install dependencies:

bash
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn
Run the Jupyter Notebook
Open the notebook in your browser:

bash
Copy code
jupyter notebook
Dataset
Ensure the file AB_NYC_2019.csv is in the same directory as your notebook.

Author 💻
Sruteka PJ
Data Science Enthusiast

LinkedIn: www.linkedin.com/in/sruteka-pj-a50a14266
GitHub: https://github.com/Sruteka

License 📜
This project is licensed under the MIT License.
