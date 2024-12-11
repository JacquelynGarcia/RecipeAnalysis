# **The Hidden Link Between Calories and Ratings**
**Author:** Jacquelyn Garcia  

---

## **Overview**  
This data science project, conducted at UCSD under the supervision of Professor Sam Lau, is aimed at understanding the relationship between calories and ratings. This analysis aims to understand whether recipes with higher calorie counts are rated differently than those with lower calorie counts. We leveraged predictive models to forecast average ratings based on recipe attributes and reported our findings.

---

## **Introduction**  
With the rise of food bloggers and self-proclaimed 'foodies', we have seen an uptick in widely available recipes online for everyone to try. Many of the most popularly shared recipes are decadent and rich foods, often associated with a high amount of calories. The United States is commonly ranked among the top countries with the highest obesity rate in the world ([World Population Review, 2024](https://worldpopulationreview.com/country-rankings/obesity-rates-by-country)). Obesity can commonly be linked to other health issues like cardiovascular diseases, diabetes, cancer, fatty liver disease, and mental health illnesses. Taking this into consideration, I will be investigating the relationship between rating and calories in a recipe. I hypothesize that people would be more willing to try and give a higher rating to a more calorically dense meal as opposed to a healthier option. In order to conduct this analysis, we will be using two datasets from [food.com](https://www.food.com) which consist of recipes and ratings. 

### Datasets

`Recipes`: Information about 83,782 unique recipes, including nutrition and preparation details.
This dataset consists of the following columns:

| Column           | Description                                                                                                                                  |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `name`           | Recipe name                                                                                                                                 |
| `id`             | Recipe ID                                                                                                                                   |
| `minutes`        | Minutes to prepare recipe                                                                                                                   |
| `contributor_id` | User ID who submitted this recipe                                                                                                           |
| `submitted`      | Date recipe was submitted                                                                                                                   |
| `tags`           | Food.com tags for recipe                                                                                                                    |
| `nutrition`      | Nutrition information in the form `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV stands for "percentage of daily value" |
| `n_steps`        | Number of steps in recipe                                                                                                                   |
| `steps`          | Text for recipe steps, in order                                                                                                             |
| `description`    | User-provided description                                                                                                                   |
| `ingredients`    | Text for recipe ingredients                                                                                                                 |
| `n_ingredients`  | Number of ingredients in recipe                                                                                                             |

`Interactions`: 731,927 user interactions, including ratings and reviews for the recipes.
This dataset's column are as follows:

| Column       | Description               |
|--------------|---------------------------|
| `user_id`    | User ID                   |
| `recipe_id`  | Recipe ID                 |
| `date`       | Date of interaction       |
| `rating`     | Rating given              |
| `review`     | Review text               |


After merging these datasets, we conducted exploratory analysis, built predictive models, and performed a fairness analysis. The most relevant columns in our analysis include:
- **Calories**: Total calorie content of the recipe.  
- **Average Rating**: Average user rating per recipe (1-5).  
- **Low Calories**: A binary feature indicating if the recipe’s calorie count is below 500.  

By investigating the relationship between calorie content and ratings, this study sheds light on how nutritional factors affect user preferences.

---

## **Data Cleaning and Exploratory Data Analysis**  
### Data Cleaning

To prepare the data for analysis, we performed several data cleaning steps that ensured the dataset was structured appropriately and contained relevant features. These steps addressed missing values, derived meaningful features, and reduced dimensionality:

1. Handling Missing Values
   - Replaced `0` values in the `rating` column with `NaN`. Ratings are typically on a scale of 1 to 5, so a `0` indicates missing data. This adjustment prevents bias in the ratings analysis and ensures calculations like averages to exclude these values from calculations.

2. Splitting Nutrition Information
   - The `nutrition` column, initially constructed as a string resembling a list, was split into individual columns: `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, and `carbohydrates`. Each column was converted to numeric types to make computations and aggregations possible.

3. Creating a Boolean Feature for Low-Calorie Recipes
   - Added a new column, `low_calories`, to classify recipes with `calories ≤ 500` as `True` and those with `calories > 500` as `False`. This feature simplifies the analysis of low-calorie recipes compared to others.

4. One-Hot Encoding of Ingredients
   - Extracted key ingredients (`butter`, `eggs`, `garlic cloves`, `milk`, `olive oil`, `onion`, `pepper`, `salt`, `sugar`, `water`) from the `ingredients` list column and converted them into one-hot encoded binary features. These features indicate whether each ingredient is present (`1`) or absent (`0`) in a recipe, enabling us to analyze the relationship between specific ingredients and ratings. Further, we wanted to see if these ingredients would prove to be relevant features once we began using our models.

5. Dropping Irrelevant or Redundant Columns
   - Removed unnecessary columns such as `description`, `tags`, and `steps`, which were not relevant to our analysis. This step reduced the size of the dataset and focused on features directly related to our research question.

6. Handling Outliers and Scaling
   - Outliers in the `calories` and `sodium` columns were examined, but no transformations were applied in this stage. Numerical features will be scaled during the modeling phase if necessary.

### Impact of Data Cleaning
These cleaning steps made the dataset more manageable and allowed us to focus on key features, such as the relationship between `calories` and `average_rating`. By creating derived features like `low_calories` and encoding ingredients, we aligned the dataset with our analysis goals. The cleaned dataset consists of the following relevant columns:

| Column              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `name`              | Recipe name                                                                |
| `id`                | Recipe ID                                                                  |
| `minutes`           | Time to prepare the recipe (in minutes)                                    |
| `n_ingredients`     | Number of ingredients in the recipe                                        |
| `average_rating`    | Average user rating (1-5)                                                  |
| `calories`          | Total calorie content of the recipe                                        |
| `low_calories`      | Boolean feature indicating low-calorie recipes (`calories ≤ 500`)          |
| `butter`            | Indicates presence of butter in the recipe (1 or 0)                       |
| `eggs`              | Indicates presence of eggs in the recipe (1 or 0)                         |
| `garlic_cloves`     | Indicates presence of garlic cloves in the recipe (1 or 0)                |
| `milk`              | Indicates presence of milk in the recipe (1 or 0)                         |
| `olive_oil`         | Indicates presence of olive oil in the recipe (1 or 0)                    |
| `onion`             | Indicates presence of onion in the recipe (1 or 0)                        |
| `pepper`            | Indicates presence of pepper in the recipe (1 or 0)                       |
| `salt`              | Indicates presence of salt in the recipe (1 or 0)                         |
| `sugar`             | Indicates presence of sugar in the recipe (1 or 0)                        |
| `water`             | Indicates presence of water in the recipe (1 or 0)                        |

Below are the data types of the columns in the cleaned dataset:

| Column            | Data Type   |
|--------------------|-------------|
| `name`            | object      |
| `id`              | int64       |
| `minutes`         | int64       |
| `n_ingredients`   | int64       |
| `rating`          | float64     |
| `average_rating`  | float64     |
| `calories`        | float64     |
| `total_fat`       | float64     |
| `sugar`           | float64     |
| `sodium`          | float64     |
| `protein`         | float64     |
| `saturated_fat`   | float64     |
| `carbohydrates`   | float64     |
| `low_calories`    | bool        |
| `butter`          | int64       |
| `eggs`            | int64       |
| `garlic cloves`   | int64       |
| `milk`            | int64       |
| `olive oil`       | int64       |
| `onion`           | int64       |
| `pepper`          | int64       |
| `salt`            | int64       |
| `sugar`           | int64       |
| `water`           | int64       |

These data types ensure that the dataset is properly formatted for analysis and modeling.

### Results
Our cleaned DataFrame contains 234,429 rows and 24 columns. Below is a preview of the first 5 rows of the cleaned DataFrame:

| name                             | id      | minutes | n_ingredients | rating | average_rating | calories | total_fat | sugar | sodium | ... | butter | eggs | garlic_cloves | milk | olive_oil | onion | pepper | salt | sugar | water |
|----------------------------------|---------|---------|---------------|--------|----------------|----------|-----------|-------|--------|-----|--------|------|---------------|------|-----------|-------|--------|------|-------|-------|
| brownies in the world best ever | 333281  | 40      | 9             | 4.0    | 4.0            | 138.4    | 10.0      | 50.0  | 3.0    | ... | 0      | 1    | 0             | 0    | 0         | 0     | 0      | 0    | 0     | 0     |
| in canada chocolate chip cookies | 453467 | 45      | 11            | 5.0    | 5.0            | 595.1    | 46.0      | 211.0 | 22.0   | ... | 0      | 1    | 0             | 0    | 0         | 0     | 0      | 1    | 0     | 1     |
| broccoli casserole               | 306168  | 40      | 9             | 5.0    | 5.0            | 194.8    | 20.0      | 6.0   | 32.0   | ... | 0      | 0    | 0             | 1    | 0         | 0     | 0      | 1    | 0     | 0     |
| broccoli casserole               | 306168  | 40      | 9             | 5.0    | 5.0            | 194.8    | 20.0      | 6.0   | 32.0   | ... | 0      | 0    | 0             | 1    | 0         | 0     | 0      | 1    | 0     | 0     |
| broccoli casserole               | 306168  | 40      | 9             | 5.0    | 5.0            | 194.8    | 20.0      | 6.0   | 32.0   | ... | 0      | 0    | 0             | 1    | 0         | 0     | 0      | 1    | 0     | 0     |


---

## Univariate Analysis
For this analysis, we examined the distribution of key features in our dataset to better understand their trends and patterns.

### **Calories**
The distribution of calories in recipes is highly skewed to the right. Most recipes on the platform have fewer than 500 calories, while only a small proportion of recipes have extremely high calorie counts. This indicates that lower-calorie recipes dominate the dataset, but there are still a few outliers with significantly higher calorie values.

![Distribution of Calories](images/calories_uni.png)

---

### **Average Rating**
The distribution of average ratings is heavily concentrated around the higher end of the scale, with most recipes having an average rating of 4 or 5. Very few recipes have ratings below 3, indicating a general bias towards positive feedback in the dataset. This trend suggests that users on the platform are more likely to rate recipes highly.

![Distribution of Average Rating](images/rating_uni.png)


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
