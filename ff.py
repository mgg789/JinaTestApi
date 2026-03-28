import requests

API_URL = "https://ai.droidje-cloud.ru"
AUTH = ("login", "password")

response = requests.post(
    f"{API_URL}/embed",
    auth=AUTH,
    json={"text": "Привет, мир!"},
)
response.raise_for_status()

embeddings = response.json()["embeddings"]  # list[list[float]]
print(embeddings[0])  # первые 5 значений вектора