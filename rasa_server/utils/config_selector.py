import json
import logging
from rasa.shared.utils.io import read_config_file

valid_langs = {
    "en": "en",
    "tw": "zh",
}

logger = logging.getLogger()  # get the root logger


def select_config(args):
    # load config templates
    lang = valid_langs.get(args["language"], args["language"])
    if lang in ["en", "zh"]:
        config_path = f"./configs/{lang}_config.yml"
        _config = read_config_file(config_path)
    else:
        config_path = "./configs/default_config.yml"
        _config = {"language": lang}
        _config.update(read_config_file(config_path))

    # pipeline
    _pipeline = {
        "name": "FallbackClassifier",
        "threshold": args["fallback_threshold"],
        "ambiguity_threshold": 0.1,
    }
    if args["fallback"] and _pipeline not in _config["pipeline"]:
        _config["pipeline"].append(_pipeline)

    for item in _config["pipeline"]:
        if item["name"] == "DIETClassifier":
            item["epochs"] = args["epochs"]

    logger.debug(f"_config: {_config}")
    return json.dumps(_config)
