import os
import random


def read_random_text_file(directory_path="articles"):
    file_list = os.listdir(directory_path)
    text_files = [file for file in file_list if file.endswith(".txt")]
    if len(text_files) == 0:
        raise FileNotFoundError(
            "Please put the article.txt under the `articles` folder"
        )
    random_file = random.choice(text_files)
    with open(os.path.join(directory_path, random_file), "r", encoding="utf-8") as file:
        file_contents = file.read()
    return file_contents
