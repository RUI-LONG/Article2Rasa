import os
import yaml
from typing import List, Dict, Text
from yaml.parser import ParserError
from yaml.scanner import ScannerError
from yamlfix import fix_code


def validate_path(func):
    def wrapper(self, path):
        if not os.path.exists(path):
            raise ValueError("Path not found")
        return func(self, path)

    return wrapper


class YamlLoader:
    @staticmethod
    def load_yaml(data):
        if isinstance(data, str):
            return yaml.safe_load(data)
        return data


class YamlFormatter(YamlLoader):
    @classmethod
    def convert_to_rasa_yaml(cls, data) -> Dict:
        data = cls.load_yaml(data)
        nlu = []

        # Loop through each intent in the YAML data
        intent_key = "intents" if data.get("intents") else "nlu"
        for intent in data[intent_key]:
            # Get the intent name
            if intent.get("intent"):
                intent_name = intent["intent"]
            else:
                intent_name = list(intent.keys())[0]

            if intent.get("examples"):
                examples = intent["examples"]
            else:
                examples = intent[intent_name]["examples"]

            nlu.append(
                {
                    "intent": intent_name,
                    "examples": examples.strip(),
                }
            )

        responses = {}
        rules = []
        intents = []

        for intent in data[intent_key]:
            if intent.get("intent"):
                intent_name = intent["intent"]
            else:
                intent_name = list(intent.keys())[0]
            intents.append(intent_name)

            if data.get("responses"):
                responses = data["responses"]
            else:
                responses[f"utter_{intent_name}"] = [{"text": intent["answer"]}]

            # Add rules
            rule = {
                "rule": f"{intent_name}_rule",
                "steps": [
                    {"intent": intent_name},
                    {"action": f"utter_{intent_name}"},
                ],
            }
            rules.append(rule)

        return intents, nlu, rules, responses


class RasaData(YamlLoader):
    def __init__(
        self,
        input_text: Text = "",
        intents: List = [],
        nlu: List = [],
        rules: List = [],
        responses: Dict = {},
    ) -> None:
        self.input_text = input_text
        self.intents = intents
        self.nlu = nlu
        self.rules = rules
        self.responses = responses

    @classmethod
    def validate_yaml(cls, input_text: Text) -> None:
        """
        Validates the input YAML text using the `yaml.safe_load` method.
        Raises an exception if the YAML is not valid.
        Args:
            input_text (str): YAML format text
        """
        try:
            input_text = (
                input_text.replace("```", "")
                .replace("yaml", "")
                .replace("- null", "")
                .replace("null", "")
                .replace("\t", " " * 4)
            )
            if input_text.count("utter") > 2:
                last_utterance = input_text.rfind("utter")
                input_text = input_text[:last_utterance]

            input_text = fix_code(input_text)
            data = cls.load_yaml(input_text)
            if not isinstance(data, dict):
                raise ValueError("Cannot parse YAML to Dict")
            return data
        except (ParserError, ScannerError) as e:
            raise ValueError(f"Invalid YAML format: {e}")

    @classmethod
    @validate_path
    def read_from_path(cls, file_path: Text):
        """
        Reads the input text file and returns its contents.
        Args:
            file_path (str): Path to the input text file
        """
        with open(file_path, "r", encoding="utf-8") as f:
            input_text = f.read()

        yaml_dict = cls.validate_yaml(input_text)
        data = YamlFormatter.convert_to_rasa_yaml(yaml_dict)
        return cls(input_text, *data)

    @classmethod
    def read_from_string(cls, input_text: Text):
        """
        Parses the input YAML text and returns Rasa YAML data.
        Args:
            input_text (str): YAML format text
        """

        yaml_dict = cls.validate_yaml(input_text)
        data = YamlFormatter.convert_to_rasa_yaml(yaml_dict)
        return cls(input_text, *data)

    def as_dict(self) -> Dict[List, Dict]:
        return {
            "version": "2.0",
            "intents": self.intents,
            "nlu": self.nlu,
            "rules": self.rules,
            "responses": self.responses,
        }

    def as_yaml_text(self) -> Text:
        return yaml.dump(
            {
                "version": "2.0",
                "intents": self.intents,
                "nlu": self.nlu,
                "rules": self.rules,
                "responses": self.responses,
            },
            allow_unicode=True,
            sort_keys=False,
        )

    def save_to_file(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.as_dict(), f, allow_unicode=True, sort_keys=False)

    def __str__(self):
        return str(self.input_text)
