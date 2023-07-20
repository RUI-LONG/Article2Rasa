import copy
from rasa.shared.utils.io import *

import logging

logger = logging.getLogger(__name__)


class NluModifier:
    def generate_rules(self):
        if self.nlu["intents"] == []:
            self.nlu["intents"] = [i["intent"] for i in self.nlu["nlu"]]

        def _add_rule(intent):
            return {
                "rule": "rule_" + intent,
                "steps": [
                    {"intent": intent},
                    {"action": "utter_" + intent},
                ],
            }

        if self.nlu["rules"] == [] and self.nlu["stories"] == []:
            for _intent in self.nlu["nlu"]:
                self.nlu["rules"].append(_add_rule(_intent["intent"]))

    def fix_missing_values(self):
        if self.nlu.get("intents") in (None, []):
            all_intents = [n["intent"] for n in self.nlu.get("nlu")]
            self.nlu["intents"] = all_intents

        for _key in ["slots", "forms"]:
            if self.nlu.get(_key) == None:
                self.nlu[_key] = {}

        for _key in [
            "actions",
            "e2e_actions",
            "stories",
            "rules",
            "entities",
            "intents",
        ]:
            if self.nlu.get(_key) == None:
                self.nlu[_key] = []


class NLUHandler(NluModifier):
    def __init__(self, add_args, decoded) -> None:
        self.args = add_args
        self.nlu = read_yaml(decoded)

    def run(self):
        self.fix_missing_values()
        self.generate_rules()

        if self.args["fallback"]:
            self.add_nlu_fallback()
        if self.args["debug"]:
            logger.debug(f"NLU: \n {dump_obj_as_yaml_to_string(self.nlu)}")
            write_yaml(self.nlu, "nlu.yml")
        logger.debug(f"NLU: \n {dump_obj_as_yaml_to_string(self.nlu)}")
        return dump_obj_as_yaml_to_string(self.nlu)

    def add_nlu_fallback(self):
        """Add a nlu fallback to data."""

        self.nlu["actions"].append("action_nlu_fallback")
        self.nlu["responses"]["utter_nlu_fallback"] = [
            {"text": self.args["fallback_msg"]}
        ]

        _rule = {
            "rule": "nlu_fallback",
            "steps": [
                {"intent": "nlu_fallback"},
                {"action": "action_nlu_fallback"},
            ],
        }
        if _rule in self.nlu["rules"]:
            self.nlu["rules"].remove(_rule)
        self.nlu["rules"].append(_rule)
