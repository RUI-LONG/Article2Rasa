# coding:utf-8
import openai
from .yaml_converter import RasaData


class GPT3Generator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        with open("./credentials/openai.key", "rb") as file:
            openai.api_key = file.read().decode()

    def _init_message_log(self):
        self.message_log = []

    def _preproccess(self, text):
        return text.replace("\n", "")

    def send_message(self):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.message_log,
            max_tokens=1024,
            top_p=1,
            presence_penalty=0.6,
            frequency_penalty=1,
            stop=None,
            temperature=0.3,
        )
        for choice in response.choices:
            if "text" in choice:
                return choice.text

        return response.choices[0].message.content

    def _append_user_prompt(self, content, sys_prompt=None, language="en"):
        sys_prompt = sys_prompt.format(language=language)
        content = self._preproccess(content)
        print("=====================")
        print("sys_prompt", sys_prompt)
        print("=====================")
        self.message_log.append({"role": "system", "content": sys_prompt})
        self.message_log.append({"role": "user", "content": content})

    def _append_sys_prompt(self, sys_prompt):
        self.message_log.append({"role": "system", "content": sys_prompt})

    def generate(self, user_input, sys_prompt=None):
        is_parsing_success = False
        if not user_input:
            return is_parsing_success, "No Input Text"
        if isinstance(user_input, dict):
            user_prompt = user_input["Answer"]
        else:
            user_prompt = user_input

        self._init_message_log()
        self._append_user_prompt(user_prompt, sys_prompt)
        gpt_text = self.send_message()
        print("generated text:")
        print(gpt_text)

        try:
            rasa_format_str = RasaData.read_from_string(gpt_text).as_yaml_text()
            is_parsing_success = True
        except:
            with open("./error_case/output.txt", "w", encoding="utf-8") as file:
                file.write(gpt_text)
            return is_parsing_success, gpt_text
        return is_parsing_success, rasa_format_str
