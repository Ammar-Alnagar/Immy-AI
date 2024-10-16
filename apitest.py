import requests

API_URL = "https://llm.criticalfutureglobal.com/api/v1/prediction/c9b49588-6fb9-493e-a86a-028964b307df"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()
    
output = query({
    "question": "Hey, how are you?",
})

print(output)