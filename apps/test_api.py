import requests

# Define the API endpoint
url = 'http://127.0.0.1:8001/predict'

# Define the input data
input_data = {
    'Amount': 100,
    'Value': 200,
    'total_transaction_amount': 300,
    'average_transaction_amount': 50,
    'transaction_count': 6,
    'std_transaction_amount': 20
}

# Send a POST request to the API
response = requests.post(url, json=input_data)

# Print the response
print(response.json())