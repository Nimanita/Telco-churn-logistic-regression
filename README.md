# Telco Customer Churn Prediction

## Project Overview
This project implements logistic regression from scratch to predict customer churn in a telecommunications company. By analyzing various customer attributes and service usage patterns, the model identifies factors that influence churn probability and helps predict which customers are at risk of leaving the service.

## Dataset
The analysis uses the Telco Customer Churn dataset, which contains information about:
- Customer demographics (gender, age, partners, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, tech support, streaming)
- Billing information (monthly charges, total charges)
- Churn status (whether the customer left the company)

## Implementation Details

### Data Preprocessing Steps
1. **Initial Data Exploration**
   - Examined data types and missing values
   - Identified unique values in categorical variables
   - Visualized the distribution of key features

2. **Data Cleaning**
   - Converted 'TotalCharges' to numeric format
   - Filled missing values (replaced NaN values with 0 for numerical columns)

3. **Feature Engineering**
   - Created interaction terms between tenure and contract types
   - Developed ratio features like average monthly spend
   - Built service aggregation features (total number of services)
   - Calculated loyalty scores combining tenure with contract type
   - Generated features showing relationship between spending patterns

4. **Data Transformation**
   - One-hot encoded categorical variables (with drop_first=True to avoid multicollinearity)
   - Applied feature scaling (standardization) to numerical features
   - Split data into training (80%) and test sets (20%)

5. **Noise Reduction**
   - Identified and removed noisy features that didn't contribute to model performance
   - Selected most influential features based on weight analysis

### Model Implementation

The project implements logistic regression from scratch with the following components:

1. **Model Functions**:
   - `compute_model_prediction`: Calculates the sigmoid function for given weights and bias
   - `compute_cost`: Implements the binary cross-entropy loss function
   - `compute_gradient_descent`: Computes gradients for weight and bias updates
   - `gradient_descent`: Updates weights and bias using calculated gradients

2. **Model Enhancements**:
   - **Regularization**: Added L2 regularization to prevent overfitting
   - `compute_cost_with_regularization`: Modified cost function with regularization term
   - `compute_gradient_with_regularization`: Modified gradient calculation with regularization
   - `gradient_descent_with_regularization`: Gradient descent with L2 penalty

3. **Model Evaluation**:
   - `evaluate_classification_model`: Calculates key metrics (accuracy, precision, recall, F1)
   - `plot_confusion_matrix`: Visualizes true vs predicted values
   - `plot_roc_curve`: Displays the ROC curve with AUC calculation
   - `plot_feature_importance`: Ranks features by their influence on predictions

### Optimization Process

1. **Hyperparameter Tuning**:
   - Tested various learning rates (alpha)
   - Experimented with different regularization strengths (lambda)
   - Adjusted the number of iterations for convergence

2. **Threshold Optimization**:
   - Evaluated model performance across different decision thresholds (0.25-0.6)
   - Analyzed precision-recall tradeoffs at each threshold
   - Selected optimal threshold to balance precision and recall

3. **Feature Selection**:
   - Identified and removed low-importance features
   - Removed features with high correlation to reduce multicollinearity
   - Selected subset of features that provided optimal performance

## Results

### Initial Model Performance
- Accuracy: 70-72%
- Precision: ~50%
- Recall: ~82%

### Final Model Performance (After Optimization)
- **Training Set**:
  - Accuracy: 74.23%
  - Precision: 50.93%
  - Recall: 80.88%
  - F1 Score: 62.50%

- **Test Set**:
  - Accuracy: 78.28%
  - Precision: 56.29%
  - Recall: 80.43%
  - F1 Score: 66.23%

### Key Insights from the Model

1. **Most Influential Features for Predicting Churn**:
   - Contract type (month-to-month contracts have higher churn)
   - Tenure (newer customers churn more frequently)
   - Payment method (specific payment methods correlate with higher churn)
   - Internet service type and additional services

2. **Feature Engineering Impact**:
   - Interaction terms between tenure and contract type significantly improved model performance
   - The ratio between monthly charges and total charges provided strong predictive signal
   - Service aggregation features helped identify service value perception

## Visualizations

The project includes several visualizations to understand the data and model performance:

1. **Data Analysis Plots**:
   - Distribution of churn
   - Tenure vs. churn
   - Monthly charges vs. churn
   - Contract type and payment method impact on churn

2. **Model Performance Visualizations**:
   - Cost vs. iteration plots showing convergence
   - Confusion matrices for training and test sets
   - ROC curves with AUC scores
   - Feature importance bar charts

## Future Improvements

To achieve the target of >80% accuracy while maintaining high recall:

1. **Advanced Feature Engineering**:
   - Create more sophisticated interaction features
   - Develop customer segmentation features
   - Extract more insights from spending patterns

2. **Model Enhancements**:
   - Implement early stopping to prevent overfitting
   - Explore different regularization methods (elastic net)
   - Develop ensemble methods combining multiple models

3. **Class Imbalance Handling**:
   - Implement class weighting in the cost function
   - Explore under/oversampling techniques
   - Investigate synthetic minority oversampling (SMOTE)

4. **Alternative Models to Compare**:
   - Decision trees with pruning
   - Random forests
   - Gradient boosting machines
   - Neural networks for complex patterns

## Setup and Usage

### Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (for preprocessing and comparison)

### Running the Code
1. Clone the repository
2. Ensure the Telco-customer-churn.csv file is in the project directory
3. Run the notebook to see the full analysis and model development process

## Conclusion

This project demonstrates how a well-implemented logistic regression model with careful feature engineering can provide valuable insights for customer churn prediction. The focus on balanced metrics (precision, recall, and accuracy) allows for practical business applications in customer retention strategies.
