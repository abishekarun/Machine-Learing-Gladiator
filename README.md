## Machine Learning Gladiator

In this mini project, we try to compare combinations of different resampling techniques like **bootstrapping**,**repeated cross validation** and **k-fold validation** along with different methods such as **up sampling**,**down sampling** and **smote**(synthetic samples) for negating the effect of imbalanced data and find out the best combination using validation set for car evaluation dataset. For this purpose, basic **random forest** algorithm is used. 

After finding out the best combination, we find out the better algorithm among **svmlinear**,**xgblinear**,**xgbtree** and **gradient boosted machine**. As expected, xg boosted algorithms perform better and xgblinear performs the best among the four.

For all comparison purposes, micro avergae F1 score and macro average F1 score is used to find out the better model/algorithm.

The Final Markdown file is [here](https://github.com/abishekarun/Machine-Learing-Gladiator/blob/master/car_evaluation.md) for this project.

The resources that helped me are:

+ [Estimate Model Accuracy in R](https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/)
+ [Caret Resampling methods](https://stats.stackexchange.com/questions/17602/caret-re-sampling-methods)
+ [The Caret Package](https://topepo.github.io/caret/) 