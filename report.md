# CS-233: Introduction to Machine Learning - Milestone 1 Report
**Project:** Gaming and Mental Health Dataset Analysis  
**Authors:** [Tes Noms/SCIPERs]  

---

## 1. Introduction
The objective of this project is to explore and predict addiction metrics from the *Gaming and Mental Health Dataset*, a synthetic dataset tracking the habits of gamers. The goal of this first milestone is to implement and evaluate baseline Machine Learning algorithms on a subset of 2000 downsampled users (1600 training, 400 test) across 13 varying features. We approach this dataset through two specific lenses: 
1. **Regression Task**: Predicting an integer continuous "addiction score" (from 0 to 10).
2. **Classification Task**: Predicting a categorical "addiction level" mapped to {0: Low, 1: Medium, 2: High}. 

To accomplish this, we implemented three distinct baseline models from scratch without the use of external Machine Learning libraries: a closed-form Linear Regression (for the regression task), an iterative Logistic Regression (for the classification task), and a K-Nearest Neighbors (KNN) model capable of accommodating both tasks.

## 2. Methodology

### 2.1 Data Preparation
Robust preprocessing was essential to ensure fairness and stability during model training. In our main script, we established two execution modes. By default, the 1600 training samples were partitioned into a localized **80% Training Set (1280 samples) and a 20% Validation Set (320 samples)**. This simulated test environment ensured that hyperparameter tuning and model selection remained devoid of data leakage, strictly reserving the provided 400 test samples for final evaluation.

A crucial preprocessing step was **Feature Normalization**. Using standard logic $Z = \frac{x - \mu}{\sigma}$, we anchored continuous features to a similar scale, effectively preventing larger-magnitude features from disproportionately dominating distance metrics (critical for KNN) or destabilizing analytical gradients (critical for Logistic Regression). A conditional check was included to prevent division by zero in instances where variance was nil. This normalization step was applied to both the training features and the validation/test features, using the mean and standard deviation strictly derived from the training data.

### 2.2 Model Implementations
- **Linear Regression**: Constructed to solve the regression task using the closed-form pseudo-inverse solution $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. A column of ones was systematically appended to the feature matrix $\mathbf{X}$ to account for the bias parameter inherently.
- **Logistic Regression**: To handle the multi-class (K=3) classification problem, we used a Softmax framework combined with Gradient Descent. A mathematical stability trick ($z - \max(z)$) was deployed to negate numerical overflows (`NaN`) within exponential calculations. Learning rates and maximum iterations were maintained as tunable hyperparameters.
- **K-Nearest Neighbors (KNN)**: The KNN was developed to predict labels based on Euclidean distance computations. Distances were obtained in a fully vectorized mathematical approach or heavily optimized loops. For classification, the algorithm applies the `np.argmax(np.bincount())` to find the most frequent categorical neighbor. For regression, it outputs the continuous mean of its $K$ nearest points.

## 3. Experiments & Results

### 3.1 Hyperparameter Tuning (Validation Phase)
To optimize the algorithms, we explored different parameter grids directly on our internal Validation set (320 samples) before evaluating them.

**KNN Hyperparameter Sweep ($K$)**
For both Regression and Classification, we swept through multiple neighbor configurations ($K \in \{1, 3, 5, 10, 15\}$). 
* For *Regression*, $K=10$ provided the lowest MSE (1.850). Lower values ($K=1$) suffered from heavy variance (MSE = 3.34).
* For *Classification*, $K=5$ maximized stability and performance, yielding an Accuracy of 76.87% and a Macro F1-Score of 0.651.

**Logistic Regression Sweep (Learning Rate & Iterations)**
We noticed that default testing hyperparameters (`lr=1e-3, max_iters=500`) produced baseline accuracy levels (~80.00%) but relatively low F1-scores (~0.500), hinting at underfitting and slow gradient convergence. Pushing the boundary to `--lr 0.1` and `--max_iters 1000` dramatically improved the Validation Accuracy to 84.37% and skyrocketed the F1-Score to 0.773, showcasing a much healthier mapping across all three categorical classes rather than simply predicting the majority class.

### 3.2 Final Evaluation on Test Set
Following the optimal hyperparameter selections, all algorithms were trained on the entire robust 1600 training samples and evaluated on the hidden 400 test samples. 

| Method | Task | Best Hyperparameters | Accuracy | F1-Score | MSE | Train / Predict Runtimes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Reg.** | Regression | *closed-form* | - | - | **0.993** | 0.002s / 0.000s |
| **KNN** | Regression | K = 10 | - | - | 1.860 | 0.130s / 0.031s |
| **Logistic Reg.** | Classification | lr=0.1, iters=1000 | **88.25%** | **0.819** | - | 0.265s / 0.000s |
| **KNN** | Classification | K = 5 | 79.00% | 0.611 | - | 0.135s / 0.036s |

## 4. Discussion & Conclusion

### 4.1 Comparative Analysis
The differences in capabilities between the parametric and non-parametric formulas proved substantially varied:
- **Regression Task**: The parametric Linear Regression drastically outperformed KNN (MSE of 0.993 vs 1.860). The linear relationship assumptions natively capture the trend of predicting "addiction scoring" far better than localized average sampling.
- **Classification Task**: Similarly, Logistic Regression dominated KNN (88.25% vs 79.00% accuracy). While an 88% accuracy is high, it is the F1-Score (0.819) that authenticates Logistic Regression's dominance, confirming that gradient optimization correctly mapped decision boundaries for all the three class levels uniformly, avoiding class imbalance traps.

### 4.2 Runtime & Efficiency
Evaluating computational time (captured via `time.time()`) offers insight on theoretical constraints. The Linear and Logistic Regressions demonstrate noticeable upfront training times (especially Logistic taking 0.265s to run 1000 iterations), but inference (prediction) is computationally instantaneous (0.0000s) as it essentially executes a single dot product (`X @ W`).
Conversely, the "lazy-learning" KNN displays a heavy contrast. It possesses effectively zero analytical mathematics to train, heavily bottlenecking on inference time (approximatively 0.035s just for 400 samples) because predicting data demands a calculation of Euclidean differences between test points and the entire dense training matrix format. 

### 4.3 Conclusion
This milestone successfully confirms that linear baseline mechanisms behave remarkably well on appropriately scaled datasets. Overall, properly optimized Logistic and Linear Regressions stand as the definitively superior choices over KNN for the Gaming and Mental Health data features, both in term of sheer metric accuracy and predictive runtime viability.
