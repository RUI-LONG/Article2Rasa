import json
import logging
from typing import Any, Text, List, Dict

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import EventType, UserUtteranceReverted, ConversationPaused

logger = logging.getLogger(__name__)


class ActionNLUFallback(Action):
    def name(self) -> Text:
        return "action_nlu_fallback"

    def read_json(self, file_name):
        with open(file_name, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
        
    def __init__(self) -> None:
        self.exclude_intents = ["nlu_fallback"]


    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List[EventType]:
        # Return nth highest intents
        nth = 3
        intent_list = [i for i in tracker.latest_message["intent_ranking"]
                       if i["name"] not in self.exclude_intents]
        buttons = list()
        high_intents = [i for i in intent_list if i["confidence"] >= 0.12]
        if len(high_intents) == 1:
            dispatcher.utter_message(response = "utter_"+high_intents[0]["name"])
        else:
            for intent in intent_list[:nth]:
                title = intent["name"][intent["name"].find("#")+1:]
                buttons.append({"title":title, "payload": "/" + intent["name"]})
                
            dispatcher.utter_message(
                response="utter_nlu_fallback", buttons=buttons)

        return []
