"""Locust load test: concurrent users hitting the sentiment API.

Run: locust -f scripts/locustfile.py --host http://localhost:8000
"""

import random

from locust import HttpUser, between, task

SAMPLE_HEADLINES = [
    "Company profits rise sharply in the third quarter",
    "Stocks fall as investors fear higher interest rates",
    "Markets remain flat ahead of the central bank decision",
    "Quarterly earnings beat analyst estimates",
    "Investors expect higher dividends this year",
]


class SentimentAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict_single(self):
        self.client.post("/predict", json={"text": random.choice(SAMPLE_HEADLINES)})

    @task(1)
    def predict_batch(self):
        self.client.post("/predict/batch", json={"texts": random.sample(SAMPLE_HEADLINES, 4)})

    @task(1)
    def health_check(self):
        self.client.get("/health")
