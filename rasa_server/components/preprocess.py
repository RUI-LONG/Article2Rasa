import typing
from rasa_sdk import logger
from .tools import Punctuation
from rasa.nlu.components import Component
from typing import Any, Optional, Text, Dict

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class DeleteSymbols(Component):
    defaults = {}
    language_list = None

    def __init__(self, component_config=None):
        super(DeleteSymbols, self).__init__(component_config)
        self.punc = Punctuation()

    def process(self, message, **kwargs):
        self.punc.set_query(message.data['text'])
        self.punc.preocess()
        message.data['text'] = self.punc.query

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component
        else:
            return cls(meta)