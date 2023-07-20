from langchain.prompts import PromptTemplate

## Use a shorter template to reduce the number of tokens in the prompt
template = """Use the provided FAQ information to create a YAML file for Rasa, with three intents, five examples, and three corresponding response texts. The responses should be concise, directly answering the intent, and written in English. Compile everything into a single YAML output. Do not add any comments or notes."""

DEFAULT_PROMPT = PromptTemplate(template=template, input_variables=[])
