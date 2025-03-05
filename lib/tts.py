import requests


def get_audio(text, api_endpoint, api_key, model):
    """Get audio from OpenAI-compatible API"""
    # Construct the proper endpoint URL for audio generation
    if api_endpoint.endswith('/v1'):
        api_endpoint = f"{api_endpoint}/audio/speech"
    elif not api_endpoint.endswith('/audio/speech'):
        api_endpoint = f"{api_endpoint.rstrip('/')}/v1/audio/speech"

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': model,
        'input': text
    }

    try:
        response = requests.post(api_endpoint, headers=headers, json=data)

        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            return None

        # Return the audio content
        return response.content
    except Exception as e:
        print(f"API request error: {e}")
        return None
