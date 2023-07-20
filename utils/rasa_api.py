import requests


def train_model(payload, user_id="user", project_id="test1"):
    url = (
        f"http://127.0.0.1:5623/model/train?user_id={user_id}&"
        + f"project_id={project_id}&epochs=100&fallback_threshold=0.5&"
        + "language=en&fallback=True"
    )

    response = requests.request(
        "POST", url, headers={"Content-Type": "application/x-yaml"}, data=payload
    )

    return response.status_code
