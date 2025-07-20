# import requests

# url = "http://127.0.0.1:5000/predict"

# samples = [
#     # Sample likely to predict class 0 (baseline)
#     [0.3, -1.1, 0.2, 1.0, -0.5, 0.6, 0.9, 1.1, -0.7, 0.3, -1.2, 1.3, 0.2, -0.3, 1.5, -0.6, 0.4, 0.8, -0.2, 0.9],
    
#     # Sample skewed toward class 1 (positive bias on selective features)
#     [-0.9, 1.3, 0.8, -0.5, 0.7, -1.2, 1.1, -0.6, 0.3, 0.4, 0.9, -1.0, 0.2, 0.6, -0.8, 1.0, -0.3, 0.7, 0.5, -0.4],

#     # Sample skewed toward class 2 (more mixed/rare pattern)
#     [1.5, 0.6, -1.2, -0.9, 0.4, 0.2, -0.6, -1.1, 0.5, -0.7, 1.0, -0.3, -0.8, 1.2, 0.3, -0.5, 0.6, -0.9, 0.1, 1.3],
# ]

# for i, features in enumerate(samples):
#     response = requests.post(url, json={"features": features})
#     print(f"Sample {i} Prediction:", response.json())
