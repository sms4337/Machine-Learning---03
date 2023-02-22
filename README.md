# Machine-Learning---03
Prediction using KNN Algorithm

1.Introduction
The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.

2.	Data Exploration: 
Let’s explore the dataset. It shows which users have purchased an Iphone. Our objective in this project is to predict if the customer will purchase an Iphone or not given their gender, age and salary. The sample rows are shown below. The column Gender is alphanumeric which is converted to numeric values. The columns “Gender”, “Age”, “Salary” are selected as features and the column “Purchase Iphone” is selected as a target.

![image](https://user-images.githubusercontent.com/21077069/220509378-91ff25c4-aabf-44ca-aabf-15d2ce27c2b9.png)

3. Load the dataset
We will assign the three independent variables “Gender”, “Age”, “Salary” to X. 
The dependent variable that we need to predict – “Purchased iPhone” will be assigned to y.
![image](https://user-images.githubusercontent.com/21077069/220509796-02148562-b4ce-4036-a044-7d16fe9344d0.png)


4. Convert Gender to Number
The classification algorithm in sklearn library cannot handle the categorical (text) data. In our data, we have the Gender variable which we have to convert to numerical format. 
We will use the class LabelEncoder to convert text data of Gender variable to number.
![image](https://user-images.githubusercontent.com/21077069/220509808-dd911956-d0f3-4008-b6c4-e71c46deb7fe.png)


5. Split the data into training and test set
We will use the train_test split method to split our data into training and testing set. We will use 25% data for testing purpose.
![image](https://user-images.githubusercontent.com/21077069/220509837-69c2a25f-4275-45c5-9f6d-29203b76bedc.png)


The train-test split is a technique for evaluating the performance of a machine learning algorithm. It can be used for classification or regression problems and can be used for any supervised learning algorithm. The procedure involves taking a dataset and dividing it into two subsets. The first subset is used to fit the model and is referred to as the training dataset. The second subset is not used to train the model; instead, the input element of the dataset is provided to the model, then predictions are made and compared to the expected values. This second dataset is referred to as the test dataset. Train Dataset: Used to fit the machine learning model. Test Dataset: Used to evaluate the fit machine learning model. The objective is to estimate the performance of the machine learning model on new data: data not used to train the model.

6. Feature Scaling and Fit the Classifier
We will be using the DecisionTreeClassifier from the sklearn.tree library. When we create the object of DecisionTreeClassifier, we will set the criterion parameter as entropy.
![image](https://user-images.githubusercontent.com/21077069/220512632-c5d01582-fb8a-418c-a1aa-30963fb3fa0d.png)

For k Nearest Neighbor algorithm also we have to do feature scaling. We will use Standard Scaler for this purpose.
We are going to use the KNeighborsClassifier class from sklearn.neighbors library. When we create an object of this class, it takes many parameters. Firstly, we have to specify the number of neighbors. In our example, let’s select 5. Then we have to define which distance method we want to use.
•	For Euclidean distance we have to specify metric as minkowski and p=2
•	For Manhattan distance we have to specify metric as minkowski and p=1
In this project we will be using the Euclidean distance.



7. Make Predictions
Now that we have trained the model, let’s make some predictions using the test dataset. 
![image](https://user-images.githubusercontent.com/21077069/220509877-8b64ad54-91b9-4887-b007-ec955e366461.png)

Precision Score: It is the percentage of predicted positive events that are actually positive.
Precision = TP / (TP + FP)
![image](https://user-images.githubusercontent.com/21077069/220512696-f2bfd3ee-72b3-4cba-a32a-045f47fc1bd9.png)

