from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Define the input data model
class InputData(BaseModel):
    Amount: float
    Value: float
    total_transaction_amount: float
    average_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float

# Define API endpoint for predictions
@app.post('/predict')
async def predict(input_data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make predictions using the loaded model
    predictions = model.predict(input_df)
    
    # Format the predictions as a response
    response = {'predictions': predictions.tolist()}
    
    return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8001)