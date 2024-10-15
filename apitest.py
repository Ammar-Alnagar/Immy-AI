import requests

def send_to_LLMinBox(user_input):
    url = "https://llm.criticalfutureglobal.com/api/chat/c9b49588-6fb9-493e-a86a-028964b307df"
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {"text": user_input}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get('response', 'No response key in the JSON response')
        else:
            print(f"Failed to get response from the API. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example input to test
    user_input = "Hello, how are you?"
    response = send_to_LLMinBox(user_input)
    if response:
        print(f"API Response: {response}")
    else:
        print("No response received from the API.")
