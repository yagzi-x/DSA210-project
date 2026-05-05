# DSA210 Project

**Name:** Yağız Çuhadar  
**Student ID:** 33930  

## Project Description
This project analyzes whether movie length has an effect on IMDb ratings.  
It uses a public movie dataset and applies data analysis and simple machine learning methods to examine the relationship between duration and ratings.
## Stage 3 Progress

For this stage, I collected and cleaned the movie dataset and analyzed the relationship between movie length and IMDb rating. After cleaning, 4790 movies remained. I used visualizations such as histograms, boxplots, and scatter plots, and performed hypothesis tests.
The results show a moderate positive relationship between runtime and IMDb rating. Longer movies tend to have higher average ratings, but genre differences also play an important role.I also tested statistical significance using t-tests and correlation tests. The results were statistically significant (p < 0.05), indicating that the relationship between runtime and IMDb rating is not due to random chance.


## Stage 4 Progress

For this stage, I applied machine learning methods to predict IMDb ratings using movie runtime, genre, and release year. I tested several regression models, including a baseline mean model, Linear Regression, Ridge Regression, Decision Tree, Random Forest, and Gradient Boosting.

The models were compared using MAE, RMSE, and R2 score. According to the results, Ridge Regression performed the best among the tested models based on MAE. This shows that movie runtime, genre, and release year provide some useful information for predicting IMDb ratings, but they are not enough to fully explain ratings by themselves.

Overall, the ML results support the earlier EDA findings. Movie length has a relationship with IMDb rating, but other factors such as genre and release year also matter. Since IMDb ratings are subjective, future work could improve the model by adding more variables such as number of votes, budget, director information, or box office revenue.
