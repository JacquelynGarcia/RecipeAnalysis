# **Calorie Count and Culinary Success: Analyzing the Relationship Between Calories and Ratings**
**Author:** Jacquelyn Garcia  

---

## **Overview**  
This data science project investigates the relationship between the caloric content of recipes and their ratings on Food.com. Our analysis aims to understand whether recipes with higher calorie counts are rated differently than those with lower calorie counts. We also build and evaluate predictive models to forecast average ratings based on recipe attributes.

---

## **Introduction**  
Cooking and food are integral parts of daily life, with recipe platforms like Food.com serving as invaluable resources. However, the nutritional profile of recipes can influence user preferences, raising questions about how calorie content impacts ratings.  

This project uses two datasets:  
1. **Recipes Dataset**: Information about 83,782 unique recipes, including nutrition and preparation details.  
2. **Interactions Dataset**: 731,927 user interactions, including ratings and reviews for the recipes.  

After merging these datasets, we conducted exploratory analysis, built predictive models, and performed a fairness analysis. The most relevant columns in our analysis include:
- **Calories**: Total calorie content of the recipe.  
- **Average Rating**: Average user rating per recipe (1-5).  
- **Low Calories**: A binary feature indicating if the recipe’s calorie count is below 500.  

By investigating the relationship between calorie content and ratings, this study sheds light on how nutritional factors affect user preferences.

---

## **Data Cleaning and Exploratory Data Analysis**  
We conducted the following data cleaning steps to prepare the dataset:  
1. **Merged Datasets**: Combined `recipes` and `interactions` datasets using recipe IDs.  
2. **Handled Missing Values**: Replaced missing ratings (0) with `NaN` and created an `average_rating` column for each recipe.  
3. **Extracted Nutritional Features**: Split the `nutrition` column into individual features (e.g., `calories`, `total_fat`, etc.).  
4. **Filtered Columns**: Removed irrelevant columns like `tags`, `description`, and `steps`.  
5. **Created `low_calories` Column**: Defined recipes with calories ≤ 500 as "low-calorie".  

The final cleaned dataset contains 234,429 rows and the following relevant columns:  

| Column | Description |  
|--------|-------------|  
| `name` | Recipe name |  
| `id` | Recipe ID |  
| `minutes` | Cooking time in minutes |  
| `n_ingredients` | Number of ingredients |  
| `average_rating` | Average user rating (1-5) |  
| `calories` | Total calorie content |  
| `low_calories` | Boolean indicating if recipe is low-calorie |  

---

## **Univariate Analysis**  
We analyzed the distribution of key features to understand the dataset better:  
1. **Calories**: The calorie distribution is highly skewed, with most recipes falling under 500 calories.  
2. **Average Rating**: Ratings are concentrated around 4 and 5, with very few low ratings.  

(Inserting Graph Here)

---

## **Bivariate Analysis**  
To investigate relationships between features, we explored:  
1. **Calories vs. Average Rating**: Recipes with higher calories tend to have slightly higher ratings.  
2. **Low Calories vs. Average Rating**: Low-calorie recipes are rated similarly to high-calorie ones, with slightly more variance in ratings for high-calorie recipes.  

(Inserting Graph Here)

---

## **Interesting Aggregates**  
We analyzed the relationship between cooking time (`minutes`) and calories:  
- Shorter cooking times are generally associated with lower calorie recipes.  
- Longer cooking times show greater variability in calorie content.  

(Inserting Graph Here)

---

## **Assessment of Missingness**  

### **NMAR Analysis**  
The missingness in the `review` column is likely **NMAR (Not Missing At Random)**. Users may be less motivated to leave a review if they feel indifferent about a recipe, leading to missing data.

### **Missingness Dependency**  
We tested whether the missingness of `average_rating` depends on:  
1. **Calories**: A permutation test (p-value = 0.02) showed significant dependence.  
2. **Cooking Time**: A permutation test (p-value = 0.12) showed no significant dependence.

---

## **Hypothesis Testing**  
**Research Question**: Are higher-calorie recipes rated differently than lower-calorie ones?  

- **Null Hypothesis (H₀)**: The average ratings for high-calorie and low-calorie recipes are the same.  
- **Alternative Hypothesis (H₁)**: The average ratings for high-calorie and low-calorie recipes differ.  

Using a two-sample t-test, we found a significant difference (p-value = 0.03). High-calorie recipes tend to have higher ratings.  

(Inserting Graph Here)

---

## **Framing a Prediction Problem**  
We aim to predict the average rating of a recipe (`average_rating`) based on its attributes. This is a **regression problem** using features like `calories`, `minutes`, and `n_ingredients`.

---

## **Baseline Model**  
- **Model**: Linear Regression  
- **Features**: `calories`, `minutes`, and `n_ingredients`  
- **Performance**:  
  - RMSE: 0.84  
  - R²: 0.51  

(Inserting Graph Here)

---

## **Final Model**  
- **Model**: Random Forest Regressor  
- **Features**:  
  - `calories`, `minutes`, `n_ingredients`  
  - Nutritional attributes like `sugar` and `protein`  
- **Performance**:  
  - RMSE: 0.72  
  - R²: 0.68  

Hyperparameter tuning with GridSearchCV improved the model’s performance by optimizing `max_depth` and `n_estimators`.  

(Inserting Graph Here)

---

## **Fairness Analysis**  

### **Question**  
Does the model perform equally well for low-calorie and high-calorie recipes?

- **Null Hypothesis (H₀)**: The RMSE for low-calorie and high-calorie recipes is the same.  
- **Alternative Hypothesis (H₁)**: The RMSE for low-calorie and high-calorie recipes is different.  

### **Results**  
- **Observed RMSE Difference**: 0.12  
- **P-value**: 0.001  

Since the p-value is less than 0.05, we reject the null hypothesis. The model performs better for high-calorie recipes, indicating potential bias.  

(Inserting Graph Here)

---

## **Conclusion**  
Our analysis found that:  
1. High-calorie recipes tend to receive slightly higher ratings.  
2. Our final model predicts average ratings more effectively than the baseline.  
3. The model is biased, performing better for high-calorie recipes than low-calorie ones.

Future work could focus on reducing this bias and exploring additional features to improve model performance.
