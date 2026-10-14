🇷🇸 [Srpski](README.md) | 🇬🇧 English

# Applying Neural Networks to Financial Fraud Prediction (Credit Card Fraud)

## 1. Introduction and Problem Description

<p align="justify">
In recent years, as technology has advanced, most people have started using credit cards for purchases and everyday needs, and the number of related frauds has grown along with it. Today almost every business, from small firms to large corporations, accepts credit cards as a payment method. Credit card fraud happens in all kinds of organisations, including the home appliance and automotive industries, banks and others. Many approaches, such as data mining and machine-learning algorithms, have been applied to identify fraudulent credit card transactions, but without significant results. There is therefore a need for effective and efficient algorithms that perform considerably better. We attempt to predict fraudulent credit card use before a transaction is even approved, using an artificial neural network.
</p>

<p align="justify">
Credit card fraud detection works as follows: the user enters the details needed to make a transaction, and the transaction should be approved only after it has been checked for any illegal activity. To achieve this, the transaction details are first passed to a verification module, which classifies the transaction as either fraudulent or legitimate. Every transaction classified as fraud is rejected. Otherwise, the transaction is approved.
</p>

<p align="justify">
The main technical challenge in financial fraud detection is that the datasets are extremely imbalanced (<i>class imbalance</i>). In real conditions, regular transactions make up over 99% of all traffic, while fraud appears in only a few tenths of a percent or a few percent of cases. Traditional machine-learning algorithms often fail in such scenarios because they minimise the overall error by simply labelling every transaction as legitimate, which yields a deceptively high accuracy. Unlike other algorithms, neural networks, thanks to their complex hidden layers and non-linear activation functions, are uniquely able to detect the subtle, hidden patterns and anomalies that describe fraud, even under severe data imbalance.
</p>

<p align="justify">
The subject of this project is the implementation of several neural network models for binary classification of transactions into legitimate and fraudulent ones. Besides building and training the networks, special focus is placed on data preprocessing, on dealing with the class imbalance mentioned above, and on hyperparameter optimisation through changes in learning rates, activation functions and optimisation algorithms. Model performance is evaluated using advanced metrics.
</p>

## 2. Project Structure

<p align="justify">
Linear algorithms we have worked with before, such as logistic regression or SVM, rely on a linear assumption. For example, we could say that the higher the transaction amount, the higher the chance that it is fraudulent.
</p>
<p align="justify">
In reality, fraud is much more subtle. A transaction does not have to be worth several hundred euros to be theft. Perpetrators might make a dozen quick transactions of 5 euros each within just a few minutes. The anomaly is not in a high value, but in a specific combination of certain values seen from unusual angles.
</p>

<p align="justify">
The main goal of this project is a comparative analysis of the performance of different neural network architectures and configurations on the collected data, in order to identify the optimal model for use in real-world banking.
</p>
<p align="justify">
Financial fraud detection requires balancing two diametrically opposed business goals: maximising security (catching as many frauds as possible) and preserving an excellent customer experience (avoiding false alarms and unjustified card blocking).
</p>
<p align="justify">
By developing four models, we analyse how different engineering methods affect this trade-off.
</p>

* Model 1: **Baseline model (scikit-learn MLPClassifier)**
* Model 2: **Basic PyTorch MLP model**
* Model 3: **Weighted PyTorch MLP model**
* Model 4: **Regularised PyTorch MLP model**

## 3. Data (Source, Structure, Analysis, Preprocessing)

<p align="justify">
The dataset used in this project was downloaded from <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank">this link</a>.
It contains transactions made with credit cards by European cardholders in September 2013. All transactions took place over two days, and 492 frauds were identified out of a total of 284,807 transactions. As mentioned in the introduction, the dataset is extremely imbalanced.
As for the attributes, it contains only numerical variables, most of which are the result of a PCA transformation applied to protect user privacy. For this reason, we cannot draw any conclusions about the users' backgrounds.
</p>

<p align="justify">
The dataset description on the platform it was downloaded from states that columns V1 to V28 are the result of a PCA (Principal Component Analysis) transformation. Put simply, PCA is a technique for compressing data, that is, reducing its dimensionality. The simplest example of what PCA does comes from everyday life. When someone photographs a tree, they move it from three dimensions into two (the photograph). Their goal is to capture the tree with as many details, shadows and shapes as possible, so that the picture is as faithful as possible even though one dimension is irreversibly lost. If a bad angle is chosen, someone looking at the photo will have a harder time telling what plant it is. PCA works with data in exactly the same way: it takes a huge number of initial columns and finds the best mathematical angle from which to compress them into fewer columns while keeping the essence and the key patterns.
</p>

<p align="justify">
In our dataset, the bank originally had very clear and detailed data about each transaction: the exact geographic location, the name of the shop, the category of purchased goods, the device's IP address, that particular customer's purchase history and so on. Since this is strictly confidential data that would violate user privacy, the bank applied the PCA algorithm. PCA took all those hundreds of confidential pieces of information, mathematically mixed them and compressed them into 28 anonymous columns (V1 to V28).
</p>

<p align="justify">
The only variables left in their original form are <b>Amount</b> and <b>Time</b>. Amount is the transaction amount, and its value varies considerably between instances (from a few to several hundred euros). Time contains the seconds elapsed between each transaction and the first transaction in the dataset. The <b>Class</b> variable is the target variable and takes the value 1 (fraud) or 0 (otherwise).
</p>

<p align="justify">
For clarity, the table below shows the final structure of the dataset attributes used to train the neural network:
</p>

<div align="center">

| Attribute | Data type | Description |
| :--- | :--- | :--- |
| **V1 - V28** | Numerical (Float) | Compressed anonymous features obtained through PCA transformation |
| **Time** | Numerical (Float) | Seconds elapsed since the first transaction in the dataset |
| **Amount** | Numerical (Float) | Monetary amount of the transaction (in euros) |
| **Class** | Binary (0 or 1) | **Target variable** (0 = Regular transaction, 1 = Fraud) |

</div>

## 4. Model Architecture

<div align="center">
<img width="800" height="340" alt="neural_network_architecture" src="https://github.com/user-attachments/assets/0d6c8399-e40f-4920-a3fd-b401796ceafc" />
  <p align="justify">
    <br>
    <b>Figure 1:</b> Detailed schematic of the implemented multilayer perceptron (MLP). The diagram gives a condensed visual overview of the layers with the exact number of neurons (30 -> 64 -> 32 -> 1).
  </p>
</div>

<p align="justify">To ensure a consistent experiment and allow a fair comparison of performance, all models trained in this study share an identical base architecture. As shown in the diagram, each neural network has an input layer of 30 neurons, two hidden layers with 64 and 32 neurons respectively, and an output layer with a single neuron responsible for binary classification.</p>

## 5. Training

### Model 1 - Baseline model (scikit-learn MLPClassifier)
<p align="justify">
Model 1 was built using the <b>MLPClassifier</b> class from the <i>scikit-learn</i> library with the parameters <code>hidden_layer_sizes=(64, 32)</code>, <code>max_iter=1000</code> and a fixed <code>random_state</code>. The Adam algorithm was used for optimisation, and training was stopped dynamically: the network converged at exactly the <b>28th iteration</b> when the <i>early stopping</i> mechanism was triggered.
</p>
<p align="justify">
The learning curve shows pronounced instability and large oscillations, caused by the extreme class imbalance and a suboptimal fixed learning rate. Despite this, the final loss value is 0.042.
</p>

### Model 2 - Basic PyTorch MLP
<p align="justify">
Moving to PyTorch gives full control over the training loop. The model was trained for <b>20 epochs</b> with mini-batches of size 1024, using the <code>BCEWithLogitsLoss</code> loss function and the Adam optimiser with a learning rate of 0.001. The learning curve shows a monotonic decrease from 0.08 to 0.0017 without sudden oscillations, which indicates stable convergence.
</p>

### Model 3 - Weighted PyTorch MLP
<p align="justify">
Identical architecture and training procedure as Model 2, with one key modification: the <code>BCEWithLogitsLoss</code> loss function receives the <code>pos_weight</code> parameter, calculated as the ratio of regular transactions to frauds in the training set (~578). This coefficient forces the optimiser to penalise the model many times more harshly for every missed fraud, compensating for the extreme class imbalance.
</p>

### Model 4 - Regularised PyTorch MLP
<p align="justify">
Model 4 keeps the weighted loss function from Model 3 and adds two regularisation techniques: <b>Dropout</b> layers with probability p=0.2 after both hidden layers, and <b>L2 regularisation</b> (<code>weight_decay=1e-4</code>) inside the Adam optimiser. The goal was to reduce the <i>overfitting</i> that caused the large number of false alarms in Model 3.
</p>

## 6. Sensitivity Analysis and Hyperparameter Optimisation

### Threshold analysis

<p align="justify">
Threshold analysis was carried out in an attempt to improve the performance of Models 3 and 4, which, despite their high recall, showed very low precision at the standard decision threshold of 0.5. The idea was that raising the threshold would make the model more selective in labelling transactions as fraud, reducing the number of false alarms while keeping recall high.
</p>
<p align="justify">
However, the results showed that even at the optimal threshold of 0.85, the precision of Model 3 does not exceed 0.15, while the F1-score stays at only 0.256. This shows that the problem is not the decision threshold but the nature of the model itself: introducing the <code>pos_weight</code> parameter shifted the decision boundary so far towards the minority class that the model structurally produces a large number of false alarms, regardless of later adjustments. Threshold tuning therefore did not deliver the desired results, which further strengthens the conclusion that <b>Model 2 is the optimal choice</b>.
</p>

### Sensitivity analysis

<p align="justify">
Sensitivity analysis was performed for Model 2 (Baseline PyTorch), as the model with the best overall performance, to gain insight into which features most influence its decisions.</p>

<p align="justify">Model 2 shows influence spread across its top features: V6, V14 and Amount are at the top of the list with absolute gradient values between 0.0008 and 0.001, while the rest of the top 10 (V27, V4, V28, V7, V21, V16, V2) declines gradually in the 0.0005–0.00065 range.
</p>

<p align="justify">
The high position of the Amount variable (3rd place) is especially significant, as it is one of only two non-anonymised, directly interpretable variables in the dataset. Its influence has a clear business justification: fraud often involves transactions with unusual amounts, so it makes sense for the model to use this signal in its decisions. A model that builds its decision on a larger number of relevant features, instead of relying on a single dominant signal, is generally more robust to noise and generalises better to unseen examples. This finding is consistent with all previously measured metrics and further supports the recommendation that Model 2 is the optimal solution for fraud detection in this context.</p>

## 7. Evaluation Results

<div align="center">

| Model | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Model 1 — Sklearn Baseline | 0.60 | 0.80 | 0.69 |
| Model 2 — PyTorch Base | 0.82 | 0.82 | 0.82 |
| Model 3 — Weighted | 0.06 | 0.91 | 0.11 |
| Model 4 — Regularised | 0.05 | 0.92 | 0.10 |

</div>

### Model 1 - Baseline model (scikit-learn MLPClassifier)
<p align="justify">The model correctly identified 78 actual frauds (True Positive) and correctly classified 56,813 regular transactions (True Negative), with 20 missed frauds (False Negative) and 51 false alarms (False Positive). The F1-score for the fraud class is <b>0.69</b> and serves as the starting point for comparison.</p>

### Model 2 - Basic PyTorch MLP
<p align="justify">With only 17 false alarms and 18 missed frauds, Model 2 achieves a uniform <b>0.82</b> for precision, recall and F1-score, a significant improvement over the Baseline. The macro average of 0.91 confirms that the model performs equally well on both classes despite the imbalance.</p>

### Model 3 - Weighted PyTorch MLP
<p align="justify">With <code>pos_weight</code>, recall jumps to <b>0.91</b> (only 9 missed frauds), but precision drops to just <b>0.06</b>, with 1,418 false alarms. The F1-score is 0.11, which makes the model impractical for real-world use.</p>

### Model 4 - Regularised PyTorch MLP
<p align="justify">Dropout and L2 regularisation did not improve the situation: recall rises to <b>0.92</b> (8 missed frauds), but precision stays at <b>0.05</b>, with 1,579 false alarms. The F1-score of 0.10 is the lowest of all models, confirming that regularisation is not an adequate solution to the extreme class imbalance in this context.</p>

## 8. Discussion

<p align="justify"> The results reveal a clear precision–recall trade-off across the four model configurations and point to several important conclusions about the nature of financial fraud detection. </p> <p align="justify"> <b>Model 2 stands out as the most statistically balanced and most practically usable solution.</b> The uniform 0.82 across all three key metrics is no coincidence: it follows directly from the model learning general fraud patterns without artificially shifting the decision boundary. The sensitivity analysis further confirms this: Model 2 bases its decisions on influence spread evenly across several relevant features (V6, V14, Amount, V27) instead of relying on a single dominant signal. Such a decision structure is more robust to noise and generalises better to unseen examples. </p> <p align="justify"> <b>Models 3 and 4 demonstrate the typical pitfall of aggressive class balancing.</b> With a <code>pos_weight</code> coefficient of ~578, the loss function becomes so focused on the minority class that the model labels almost every transaction as suspicious. Threshold analysis over the 0.1–0.9 range showed that even at the optimal threshold of 0.85, the precision of Model 3 does not exceed 0.15, proving that the problem lies not in the choice of threshold but in the model's fundamental configuration. Regularisation (Dropout + L2) in Model 4 failed to correct this imbalance, because the problem is structural rather than one of generalisation. </p> <p align="justify"> <b>An important limitation of the project</b> concerns the interpretability of the results: variables V1–V28 are anonymised through PCA, so the findings of the sensitivity analysis cannot be given full business context. The only directly interpretable signal, the Amount variable, ranks third by influence in Model 2, which is consistent with well-known financial fraud patterns. </p> <p align="justify"> <b>Possible improvements</b> outside the scope of this project include: applying SMOTE to synthetically generate minority-class samples, introducing a validation set to monitor generalisation during training, and exploring advanced architectures such as LSTM networks that could exploit the temporal structure of the data (the Time variable). </p>

## 9. Conclusion

<p align="justify">
In this project, four multilayer perceptron configurations were implemented and evaluated for credit card fraud detection, with a focus on the extreme class imbalance (only ~0.17% fraud in the dataset).
</p>
<p align="justify">
<b>Model 2 (basic PyTorch MLP)</b> proved to be the optimal solution, with a uniform F1-score of 0.82 for the fraud class, the fewest false alarms (17) and a stable learning curve. The sensitivity analysis confirms that the model makes decisions based on several relevant signals, which makes it robust and reliable. Threshold analysis on Models 3 and 4 showed that even tuning the decision threshold cannot compensate for the structural problem of aggressive class balancing, which further confirms the superiority of Model 2.
</p>
<p align="justify">
The key lesson of this project is that in domains with extreme class imbalance, a more sophisticated technique does not necessarily mean a better result. A model that learns the boundaries between classes in a minimalist and stable way, without aggressive interventions in the loss function, proved superior to models with explicit balancing mechanisms. A precision of 0.82 and a recall of 0.82 make Model 2 directly applicable in a real banking environment, where catching fraud and not blocking legitimate customers are equally important.
</p>

## Authors

* Anja Perović
* Una Ilić
