# Personalized Restaurant Recommendation System with Dietary Constraints

## Team Members
* Gul Nayab (gul.nayab@sjsu.edu)

## Problem Statement

Traditional restaurant recommendation systems primarily rely on ratings or simple filtering mechanisms, but these approaches often fail to account for real-world user constraints such as dietary restrictions, travel distance, and cuisine preferences, as well as fail to offer suggestions without manual research on the user's part.

This project aims to develop a personalized restaurant recommendation system that ranks nearby restaurants based on predicted user satisfaction while respecting constraints such as:

* Location and maximum travel distance
* Cuisine preferences
* Price range
* Dietary restrictions (e.g., vegan, vegetarian, halal, gluten-free)

The goal is to design a learning-to-rank machine learning system that integrates multiple data sources and produces meaningful, explainable recommendations.

---

## Dataset / Data Source

This project uses the Yelp Open Dataset, which provides:

* Business metadata (location, categories, price range)
* User reviews and ratings
* Geographic information (latitude and longitude)

Additional features are derived from the dataset, including:

* Sentiment scores from reviews
* Frequency of dietary-related keywords (e.g., "vegan", "halal") to determine dietary restriction accomodation
* Distance between user and restaurant (computed using the Haversine formula)

User profiles are simulated and include constraints such as dietary needs, preferred cuisines, and travel limits.

---

## Planned Model / System Approach

The system is modeled as a supervised learning-to-rank problem, where restaurants are ranked based on relevance to a user profile.

### Models to be implemented:

* Baseline: Logistic Regression
* Gradient Boosting: XGBoost
* Learning-to-Rank: LambdaMART

### Feature Engineering:

* Restaurant attributes (rating, price, categories)
* Geographic distance from user
* Text-based features from reviews (sentiment + keyword relevance)
* Dietary compatibility indicators

### Pipeline:

1. Input user profile (location, preferences, constraints, previous recommendation satisfation)
2. Retrieve nearby restaurants
3. Compute feature vectors for each restaurant
4. Apply trained ranking model
5. Output top-ranked recommendations

### Explainability:

Each recommendation will include:

* Relevance to dietary restrictions
* Distance impact
* Rating contribution

---

## Current Implementation Progress

* Project proposal and system design completed
* Dataset selected (Yelp Open Dataset)
* Initial repository structure created
* Feature Engineering and Model Implementation in progress