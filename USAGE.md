# Embeddings API — как использовать

**Base URL:** `http://localhost:47821`

## Установка

```bash
pip install requests numpy
```

## Быстрый старт

```python
import requests

API_URL = "http://localhost:47821"
AUTH = ("admin", "your_password")

response = requests.post(
    f"{API_URL}/embed",
    auth=AUTH,
    json={"text": "Привет, мир!"},
)
response.raise_for_status()

embeddings = response.json()["embeddings"]  # list[list[float]]
print(embeddings[0][:5])  # первые 5 значений вектора
```

---

## Эндпоинты

### `POST /embed` — получить эмбеддинги

**Параметры запроса:**

| Поле | Тип | Обязательно | По умолчанию | Описание |
|---|---|---|---|---|
| `text` | `str` или `list[str]` | да | — | Одна строка или массив строк |
| `task` | `str` | нет | `"text-matching"` | Тип задачи (см. ниже) |
| `batch_size` | `int` | нет | `16` | Размер батча при инференсе |

**Доступные задачи (`task`):**

| Значение | Когда использовать |
|---|---|
| `"text-matching"` | Семантическое сходство, дефолт |
| `"retrieval.query"` | Запрос в поиске (вопрос) |
| `"retrieval.passage"` | Документы для индексации |
| `"classification"` | Классификация текста |
| `"separation"` | Кластеризация |

**Ответ:**

```json
{
  "embeddings": [[0.12, -0.34, ...], [0.56, 0.78, ...]],
  "shape": [2, 1024]
}
```

---

## Примеры

### Одна строка

```python
response = requests.post(
    f"{API_URL}/embed",
    auth=AUTH,
    json={"text": "Машинное обучение — это круто"},
)
vector = response.json()["embeddings"][0]  # list[float], длина 1024
```

### Массив строк

```python
texts = [
    "Первый документ",
    "Второй документ",
    "Третий документ",
]

response = requests.post(
    f"{API_URL}/embed",
    auth=AUTH,
    json={"text": texts},
)

data = response.json()
embeddings = data["embeddings"]  # list[list[float]]
print(data["shape"])             # [3, 1024]
```

### С выбором задачи

```python
# Индексируем документы
response = requests.post(
    f"{API_URL}/embed",
    auth=AUTH,
    json={
        "text": ["Документ 1", "Документ 2"],
        "task": "retrieval.passage",
    },
)
doc_embeddings = response.json()["embeddings"]

# Кодируем поисковый запрос
response = requests.post(
    f"{API_URL}/embed",
    auth=AUTH,
    json={
        "text": "что такое нейросеть",
        "task": "retrieval.query",
    },
)
query_embedding = response.json()["embeddings"][0]
```

### Получить numpy array

```python
import numpy as np

response = requests.post(
    f"{API_URL}/embed",
    auth=AUTH,
    json={"text": texts},
)
embeddings = np.array(response.json()["embeddings"])
print(embeddings.shape)  # (3, 1024)
```

### Обёртка-хелпер

```python
import numpy as np
import requests
from typing import Union

class EmbeddingsClient:
    def __init__(self, url: str, user: str, password: str):
        self.url = url.rstrip("/")
        self.auth = (user, password)

    def embed(
        self,
        text: Union[str, list[str]],
        task: str = "text-matching",
        batch_size: int = 16,
    ) -> np.ndarray:
        response = requests.post(
            f"{self.url}/embed",
            auth=self.auth,
            json={"text": text, "task": task, "batch_size": batch_size},
            timeout=600,
        )
        response.raise_for_status()
        return np.array(response.json()["embeddings"])


# Использование
client = EmbeddingsClient("http://localhost:47821", "admin", "your_password")

vec = client.embed("Привет!")                          # shape: (1, 1024)
vecs = client.embed(["Текст 1", "Текст 2"])            # shape: (2, 1024)
vecs = client.embed(["..."], task="retrieval.passage") # shape: (1, 1024)
```

---

## Важно

- Запросы выполняются **последовательно** — если отправить несколько одновременно, они встанут в очередь. Это нормально, не нужно делать retry при долгом ответе.
- Таймаут клиента выставляй не менее **600 секунд** — длинные тексты или большие батчи могут считаться долго.
- `/health` доступен без авторизации, удобно для проверки что сервер живой:

```python
requests.get(f"{API_URL}/health").json()
# {"status": "ok", "model_loaded": true}
```
