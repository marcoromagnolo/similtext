from settings import STORAGE_DIR, OPENAI_SECRET
import requests


def compare_texts(logger, text1, text2):

            # Create headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_SECRET}"
                }

            # Create body
            data = {
                "model": 'gpt-3.5-turbo',
                "messages": [
                        {"role": "system", "content": "You are a professional journalist evaluating the similarity of two articles. Two articles are considered similar if they discuss the same topic, cite the same sources or protagonists, and present the same informations."},
                        {'role': 'user', 'content': f"This is the first article: {text1}."},
                        {'role': 'user', 'content': f"This is the second article: {text2}."},
                        {'role': 'user', 'content': f"Reply with 'true' if articles are similar otherwise reply with 'false'."},
                    ],
                "temperature": 0,
                "response_format": {"type": "text"}
            }

            if call_openapi(logger, headers, data):
                data = {
                    "model": 'gpt-3.5-turbo',
                    "messages": [
                            {"role": "system", "content": "You are a professional journalist and you have compare two articles and indicate whether they come from the same source, although the writing style may be different."},
                            {'role': 'user', 'content': f"This is the first article: {text1}."},
                            {'role': 'user', 'content': f"This is the second article: {text2}."},
                            {'role': 'user', 'content': "Reply with 'true' if two articles share the same source otherwise reply with 'false'."}
                        ],
                    "temperature": 0,
                    "response_format": {"type": "text"}
                }
                return call_openapi(logger, headers, data)
            
            return False


def call_openapi(logger, headers, data):
        logger.info(f"Request: {str(data)}")
        url = "https://api.openai.com/v1/chat/completions"
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()  # Decodifica la risposta in formato JSON
            out = result["choices"][0]["message"]["content"]
            logger.info(f"Response: {out}")
            return out in {"true", "True", "TRUE", "1", "yes", "y"}

        else:
            logger.info(f"Errore: {response.status_code}")
            logger.info(response.text)
            
        return False