#  Used Car Price Estimator

A **machine learning web application** that predicts the estimated **selling price of a used car** using **Ridge Regression**, built with **Flask**, **scikit-learn**, and **pandas**.

---

##  Features

- Predicts car price based on input features
- Built with Ridge Regression model
- Web app built using Flask
- Responsive and user-friendly UI
- Clean and modern CSS styling

---

##  Machine Learning Details

- **Model**: Ridge Regression
- **Input Features**:
  - Year
  - Present Price
  - Kms Driven
  - Fuel Type
  - Seller Type
  - Transmission
  - Owner
- **Target**: Selling Price
- **Preprocessing**: One-Hot Encoding for categorical variables
- **Model File**: `ridge_model.pkl`

---

## 📁 Project Structure
```
used-car-price-estimator/
│
├── static/
│ └── style.css # CSS styling for the UI
│
├── templates/
│ ├── index.html # Input form page
│ └── result.html # Result display page
│
├── used_car.csv # Sample dataset
├── model_train.py # Script to train and save the model
├── ridge_model.pkl # Saved Ridge Regression model
├── app.py # Flask web server
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

##  How to Run the Project Locally

1. Clone the Repository

```
git clone https://github.com/yourusername/used-car-price-estimator.git
cd used-car-price-estimator
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Train the Model
If ridge_model.pkl is not already present:
```
python model_train.py
```
4. Run the Flask App
```
python app.py
```
5. Open in Browser
Go to: http://127.0.0.1:5000

## Screenshots
 
 ![alt text](<Screenshot 2025-08-01 230222.png>)
 ![alt text](<Screenshot 2025-08-01 230250.png>)
 ![alt text](<Screenshot 2025-08-01 230505.png>)