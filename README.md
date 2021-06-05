# TSF-TASK

## Task - Prediction using Unsupervised ML
From the given ‘Iris’ [dataset](https://bit.ly/3kXTdox), predict the optimum number of clusters and represent it visually.

cluster graph
![](data:image/png;base64,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%0A)
