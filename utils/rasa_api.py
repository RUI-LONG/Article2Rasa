import json
import requests


def train_model(payload, user_id="user", project_id="test1"):
    url = (
        f"http://127.0.0.1:5623/model/train?user_id={user_id}&"
        + f"project_id={project_id}&epochs=100&fallback_threshold=0.5&"
        + "language=en&fallback=True"
    )
    print("=====================")
    print(payload)

    response = requests.request(
        "POST", url, headers={"Content-Type": "application/x-yaml"}, data=payload
    )

    return response.status_code


def activate_model(user_id="user", project_id="test1"):
    url = "http://127.0.0.1:5623/model"
    payload = json.dumps({"model_file": f"{user_id}/{project_id}.tar.gz"})
    headers = {"Content-Type": "application/json"}
    response = requests.request("PUT", url, headers=headers, data=payload)
    return response.status_code
