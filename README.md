# TMDB 5000 Movies - Revenue Estimator and Movie Recommender Systems: Project Overview 
* Developed a model that estimates movie revenue (MAE to transformed revenue ~ 31.80) based on movie characteristics  
* Programmed six different types of recommender systems using the principles of demographic filtering, content based filtering, and collaborative filtering  
* Cleaned over 4800 rows of data, engineered features from the movie cast and crew lists to get the top 30 highest grossing actors, top 100 highest grossing directors, top 100 highest grossing composers, and top 100 highest grossing production companies
* Optimized Linear, Lasso, Elastic Net, SVR, Gradient Boosting, XGB and Random Forest Regressors using GridsearchCV to reach the best model (MAE ~ 32.20)
* Developed a hybrid model using 4 different models to make the final predictive model (MAE ~ 31.80) more robust to overfitting than any single baseline model

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, surprise, matplotlib, seaborn  
**Code and Information on Different Recommender Systems:**  
* https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system  
* https://medium.com/sfu-cspmp/recommendation-systems-user-based-collaborative-filtering-using-n-nearest-neighbors-bf7361dc24e0  
* https://www.bluepiit.com/blog/classifying-recommender-systems/#:~:text=There%20are%20majorly%20six%20types,system%20and%20Hybrid%20recommender%20system  

**Datasets**: 
* The original datasets 'tmdb_5000_credits.csv' and 'tmdb_5000_movies.csv' were too large to upload to github. You can find them here: https://www.kaggle.com/tmdb/tmdb-movie-metadata  
* The original dataset 'ratings_small.csv' can be found here: https://www.kaggle.com/rounakbanik/the-movies-dataset  


## Data Cleaning
After merging the 'tmdb_5000_credits.csv' and 'tmdb_5000_movies.csv' dataframes, we have the following variables:  
*	Budget  
*	Genres
*	Keywords  
*	Original Language  
*	Title  
*	Overview  
*	Popularity  
*	Production Companies  
*	Production Countries  
*	Release Date  
*	Revenue  
*	Runtime  
*	Spoken Languages  
*	Tagline  
* Vote Average  
* Vote Count


I made the following changes and created the following variables:
*	Parsed genre names, keywords, production company names, production country names, spoken languages (all originally messy strings) into lists of strings  
*	Separated release date column into year, month and date columns  
*	Parsed the cast and crew columns for the first 5 actors' names mentioned, and first mentioned executive producer, director, screenplay writer and original music composer
* Made a column for length of movie overview  
* Made a column for number of spoken languages  
* Made columns for if movies contained some of the highest grossing cast/crew members or were produced in some of the most highest grossing locations:  
    * Top 30 actors  
    * Top 100 directors  
    * Top 100 executive producers    
    * Top 100 screenplays  
    * Top 100 composers  
    * Top 100 production companies  
    * Top 15 production countries    


## EDA
I looked at the distributions and correlations of the numeric predictor variables. I used wordclouds and barplots to see the most common occurences in categorical variables. 
I finally used pivot tables to examine the relationships between our predictor variables and revenue.  

## Model Building 

First, I normalized the response variable 'revenue' using box-cox transformation with lambda = 0.204. I also imputed missing values where budget = 0 and for runtime = 0.  

Then, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 30%.   

I tried eight different models and evaluated them using Mean Absolute Error.   

I then used a mixture of the eight different models to arrive at a robust model with a decently low MAE.


## Model performance
**Baseline Models**: The random forest model performed the best
*	**Linear Regression**: MAE = 33.62  
* **Lasso Regression**: MAE = 33.63  
* **Elastic Net Regression**: MAE = 33.62  
* **Support Vector Regression**: MAE = 49.90  
* **Gradient Boosting**: MAE = 32.65  
* **Extreme Gradient Boosting**: MAE = 32.71  
* **Random Forest Regression**: MAE = 32.20  

**Hybrid Models**: The second hybrid model performed the best
* **Mixture of linear, lasso, elastic net, gradient boosting, extreme gradient boosting, random forest**: MAE = 36.28  
* **Mixture of linear, elastic net, gradient boosting, random forest**: MAE = 31.80  


## Movie Recommender Systems  

**Demographic Filtering**: I created 2 different recommender systems using 2 different scoring systems:  
* **IMDB's Weighted Rating (WR) Formula**  
* **Popularity (calculated by TMDB and already included in dataset)**  

**Content Based Filtering**: I created 2 different recommender systems using 2 different sets of metadata:  
* **Overview**  
* **Cast, Crew, Genres and Keywords**  
I then calculated their cosine similarity matrices and designed a function that uses the cosine similarity matrix to find and recommend the most similar movies to an input movie  

**Collaborative Filtering**: I explored the 2 different types of CF:  
* **User-based CF**: Created a ratings matrix for all users, calculated the cosine similarity matrix, then designed a function that uses the concept of knn neighbors to recommend the top rated movies out of those rated by input user's k nearest neighbors  
* **Item-based CF**: Designed a function that uses SVD to estimate input user's ratings for unrated movies, then recommend the movies with the top predicted user rating  





