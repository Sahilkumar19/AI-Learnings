import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

GEMINI_API_URL = "https://gemini.googleapis.com/v1alpha/models/gemini:predict"
GEMINI_API_KEY = "AIzaSyBbhN_vb0Rob5UHbGPlUO-OCpPXH31qbBI"

def preprocess_data(data):
    """
    Preprocesses the campaign data by handling missing values and normalizing metrics.
    """
    data.fillna(0, inplace=True)
    numeric_cols = ['Impressions', 'Clicks', 'Conversions', 'Spend', 'Revenue']
    data[numeric_cols] = data[numeric_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return data

def calculate_metrics(data):
    """
    Calculates key metrics: CTR, ROAS, and CPA.
    """
    data['CTR'] = (data['Clicks'] / data['Impressions']) * 100
    data['CPA'] = data['Spend'] / data['Conversions'].replace(0, 1)
    data['ROAS'] = data['Revenue'] / data['Spend'].replace(0, 1)
    return data

def optimization_rules(data, target_cpa):
    """
    Apply optimization rules for pausing, increasing, or decreasing budgets.
    """
    data['Action'] = 'No Change'

    # Pause campaigns
    data.loc[(data['CTR'] < 1) & (data['Status'] == 'Active'), 'Action'] = 'Pause'
    data.loc[(data['CPA'] > 3 * target_cpa) & (data['Status'] == 'Active'), 'Action'] = 'Pause'

    # Increase budgets
    data.loc[(data['ROAS'] > 4), 'Action'] = 'Increase Budget'
    data.loc[(data['Conversions'] > 0.2 * data['Conversions'].shift(7)), 'Action'] = 'Increase Budget'

    # Decrease budgets
    data.loc[(data['ROAS'] < 1.5), 'Action'] = 'Decrease Budget'
    return data

def sanitize_data(data):
    """
    Sanitizes the campaign data to ensure there are no NaN or infinite values.
    """
    for column in ['Impressions', 'Clicks', 'Conversions', 'Spend', 'Revenue', 'CTR', 'CPA', 'ROAS']:
        data[column] = data[column].replace([np.inf, -np.inf], 0)  # Replace infinities with 0
        data[column] = data[column].fillna(0)  # Replace NaN values with 0
    return data

def gemini_insights(data):
    """
    Use Gemini API to analyze campaign data and provide insights.
    """
    # Sanitize data before sending to the API
    data = sanitize_data(data)
    
    # Prepare the request payload
    payload = {
        "instances": [
            {
                "campaign_data": data.to_dict(orient="records")
            }
        ],
        "parameters": {
            "model": "gemini",
            "temperature": 0.7,
            "max_output_tokens": 500
        }
    }

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GEMINI_API_URL, json=payload, headers=headers)

    # Handle response
    if response.status_code == 200:
        result = response.json()
        return result.get("predictions", [{}])[0].get("output", "No insights available")
    else:
        return f"Error: {response.status_code}, {response.text}"

def visualize_data(data):
    """
    Visualize key metrics like ROAS, CTR, and budget trends.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['CampaignID'], data['ROAS'], label='ROAS')
    plt.plot(data['CampaignID'], data['CTR'], label='CTR')
    plt.legend()
    plt.title('Campaign Metrics')
    plt.xlabel('CampaignID')
    plt.ylabel('Metrics')
    plt.show()

def generate_report(data):
    """
    Generate a report summarizing actions and recommendations.
    """
    summary = data.groupby('Action').size()
    return f"Daily Summary:\n{summary}\n\nDetails:\n{data[['CampaignID', 'Action', 'CTR', 'ROAS', 'CPA']]}"

def marketing_automation_pipeline(input_csv, target_cpa):
    """
    Complete automation pipeline for AI-driven marketing optimization.
    """
    # Step 1: Load Data
    data = pd.read_csv(input_csv)

    # Step 2: Preprocess Data
    data = preprocess_data(data)

    # Step 3: Calculate Metrics
    data = calculate_metrics(data)

    # Step 4: Apply Optimization Rules
    data = optimization_rules(data, target_cpa)

    # Step 5: Get AI Insights from Gemini
    insights = gemini_insights(data)

    # Step 6: Generate Report
    report = generate_report(data)

    # Step 7: Visualize Metrics
    visualize_data(data)

    # Save Output
    data.to_csv("optimized_campaigns.csv", index=False)

    return report, insights

if __name__ == "__main__":
    INPUT_CSV = "campaign_data.csv"
    TARGET_CPA = 50

    report, insights = marketing_automation_pipeline(INPUT_CSV, TARGET_CPA)

    print("\nOptimization Report:\n", report)