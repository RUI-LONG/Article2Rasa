import os
import yaml
from cmath import log
import logging
import datetime
import multiprocessing
import traceback
from collections import defaultdict
from functools import reduce, wraps
from pathlib import Path
from http import HTTPStatus
from typing import (
    Any,
    List,
    Optional,
    Text,
    Union,
    Dict,
)

from sanic import Sanic, response
from sanic.request import Request
from sanic.response import HTTPResponse
from sanic_cors import CORS
from sanic_jwt import Initialize, exceptions

import rasa
import rasa.core.utils
import rasa.utils.common
import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.utils.endpoints
import rasa.utils.io
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.formats import RasaYAMLReader
from rasa import model
from rasa.constants import DEFAULT_RESPONSE_TIMEOUT, MINIMUM_COMPATIBLE_VERSION
from rasa.shared.constants import (
    DOCS_URL_TRAINING_DATA,
    DOCS_BASE_URL,
    DEFAULT_SENDER_ID,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_CONVERSATION_TEST_PATH,
    TEST_STORIES_FILE_PREFIX,
)
from rasa.shared.core.domain import InvalidDomain, Domain
from rasa.core.agent import Agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    OutputChannel,
    UserMessage,
)
import rasa.shared.core.events
from rasa.shared.core.events import Event
from rasa.core.lock_store import LockStore
from rasa.core.test import test
from rasa.core.tracker_store import TrackerStore
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.core.utils import AvailableEndpoints
from rasa.nlu.emulators.no_emulator import NoEmulator
import rasa.nlu.test
from rasa.nlu.test import CVEvaluationResult
from rasa.utils.endpoints import EndpointConfig
from utils.config_selector import select_config
from utils.nlu_handler import NLUHandler
from rasa.server import *

# Don't import create_app from rasa.server
del globals()["create_app"]
"""
This File is modified from the original file in rasa.server.
https://github.com/RasaHQ/rasa/blob/78a06f3edc90f6b9ae0b21d1e2c4c2fd750dfb70/rasa/server.py

Modified: $509 - 513
"""

logger = logging.getLogger(__name__)

JSON_CONTENT_TYPE = "application/json"
YAML_CONTENT_TYPE = "application/x-yaml"

OUTPUT_CHANNEL_QUERY_KEY = "output_channel"
USE_LATEST_INPUT_CHANNEL_AS_OUTPUT_CHANNEL = "latest"
EXECUTE_SIDE_EFFECTS_QUERY_KEY = "execute_side_effects"


def create_main_app(
    agent: Optional["Agent"] = None,
    cors_origins: Union[Text, List[Text], None] = "*",
    auth_token: Optional[Text] = None,
    response_timeout: int = DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_method: Text = "HS256",
    endpoints: Optional[AvailableEndpoints] = None,
) -> Sanic:
    """Class representing a Rasa HTTP server."""
    app = Sanic(__name__)
    app.config.RESPONSE_TIMEOUT = response_timeout
    configure_cors(app, cors_origins)

    # Setup the Sanic-JWT extension
    if jwt_secret and jwt_method:
        # since we only want to check signatures, we don't actually care
        # about the JWT method and set the passed secret as either symmetric
        # or asymmetric key. jwt lib will choose the right one based on method
        app.config["USE_JWT"] = True
        Initialize(
            app,
            secret=jwt_secret,
            authenticate=authenticate,
            algorithm=jwt_method,
            user_id="username",
        )

    app.agent = agent
    # Initialize shared object of type unsigned int for tracking
    # the number of active training processes

    @app.exception(ErrorResponse)
    async def handle_error_response(
        request: Request, exception: ErrorResponse
    ) -> HTTPResponse:
        return response.json(exception.error_info, status=exception.status)

    @app.get("/")
    async def hello(request: Request) -> HTTPResponse:
        """Check if the server is running and responds with the version."""
        return response.text("Hello from Developer \n")

    @app.get("/version")
    async def version(request: Request) -> HTTPResponse:
        """Respond with the version number of the installed Rasa."""

        return response.json(
            {
                "version": rasa.__version__,
                "minimum_compatible_version": MINIMUM_COMPATIBLE_VERSION,
            }
        )

    @app.get("/status")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def status(request: Request) -> HTTPResponse:
        """Respond with the model name and the fingerprint of that model."""

        return response.json(
            {
                "model_file": app.agent.path_to_model_archive
                or app.agent.model_directory,
                "fingerprint": model.fingerprint_from_path(app.agent.model_directory),
            }
        )

    @app.get("/conversations/<conversation_id:path>/tracker")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def retrieve_tracker(request: Request, conversation_id: Text) -> HTTPResponse:
        """Get a dump of a conversation's tracker including its events."""
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        until_time = rasa.utils.endpoints.float_arg(request, "until")

        tracker = await app.agent.create_processor().fetch_tracker_with_initial_session(
            conversation_id
        )

        try:
            if until_time is not None:
                tracker = tracker.travel_back_in_time(until_time)

            state = tracker.current_state(verbosity)
            return response.json(state)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/conversations/<conversation_id:path>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def append_events(request: Request, conversation_id: Text) -> HTTPResponse:
        """Append a list of events to the state of a conversation."""
        validate_events_in_request_body(request)

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.agent.lock_store.lock(conversation_id):
                processor = app.agent.create_processor()
                events = _get_events_from_request_body(request)

                tracker = await update_conversation_with_events(
                    conversation_id, processor, app.agent.domain, events
                )

                output_channel = _get_output_channel(request, tracker)

                if rasa.utils.endpoints.bool_arg(
                    request, EXECUTE_SIDE_EFFECTS_QUERY_KEY, False
                ):
                    await processor.execute_side_effects(
                        events, tracker, output_channel
                    )

                app.agent.tracker_store.save(tracker)

            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    def _get_events_from_request_body(request: Request) -> List[Event]:
        events = request.json

        if not isinstance(events, list):
            events = [events]

        events = [Event.from_parameters(event) for event in events]
        events = [event for event in events if event]

        if not events:
            rasa.shared.utils.io.raise_warning(
                f"Append event called, but could not extract a valid event. "
                f"Request JSON: {request.json}"
            )
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Couldn't extract a proper event from the request body.",
                {"parameter": "", "in": "body"},
            )

        return events

    @app.put("/conversations/<conversation_id:path>/tracker/events")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def replace_events(request: Request, conversation_id: Text) -> HTTPResponse:
        """Use a list of events to set a conversations tracker to a state."""
        validate_events_in_request_body(request)

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.agent.lock_store.lock(conversation_id):
                tracker = DialogueStateTracker.from_dict(
                    conversation_id, request.json, app.agent.domain.slots
                )

                # will override an existing tracker with the same id!
                app.agent.tracker_store.save(tracker)

            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.get("/conversations/<conversation_id:path>/story")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    @ensure_conversation_exists()
    async def retrieve_story(request: Request, conversation_id: Text) -> HTTPResponse:
        """Get an end-to-end story corresponding to this conversation."""
        until_time = rasa.utils.endpoints.float_arg(request, "until")
        fetch_all_sessions = rasa.utils.endpoints.bool_arg(
            request, "all_sessions", default=False
        )

        try:
            stories = get_test_stories(
                app.agent.create_processor(),
                conversation_id,
                until_time,
                fetch_all_sessions=fetch_all_sessions,
            )
            return response.text(stories)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/conversations/<conversation_id:path>/execute")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    @ensure_conversation_exists()
    async def execute_action(request: Request, conversation_id: Text) -> HTTPResponse:
        request_params = request.json

        action_to_execute = request_params.get("name", None)

        if not action_to_execute:
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Name of the action not provided in request body.",
                {"parameter": "name", "in": "body"},
            )

        policy = request_params.get("policy", None)
        confidence = request_params.get("confidence", None)
        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.agent.lock_store.lock(conversation_id):
                tracker = await (
                    app.agent.create_processor().fetch_tracker_and_update_session(
                        conversation_id
                    )
                )

                output_channel = _get_output_channel(request, tracker)
                await app.agent.execute_action(
                    conversation_id,
                    action_to_execute,
                    output_channel,
                    policy,
                    confidence,
                )

        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

        state = tracker.current_state(verbosity)

        response_body: Dict[Text, Any] = {"tracker": state}

        if isinstance(output_channel, CollectingOutputChannel):
            response_body["messages"] = output_channel.messages

        return response.json(response_body)

    @app.post("/conversations/<conversation_id:path>/trigger_intent")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def trigger_intent(request: Request, conversation_id: Text) -> HTTPResponse:
        request_params = request.json

        intent_to_trigger = request_params.get("name")
        entities = request_params.get("entities", [])

        if not intent_to_trigger:
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Name of the intent not provided in request body.",
                {"parameter": "name", "in": "body"},
            )

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        try:
            async with app.agent.lock_store.lock(conversation_id):
                tracker = await (
                    app.agent.create_processor().fetch_tracker_and_update_session(
                        conversation_id
                    )
                )
                output_channel = _get_output_channel(request, tracker)
                if intent_to_trigger not in app.agent.domain.intents:
                    raise ErrorResponse(
                        HTTPStatus.NOT_FOUND,
                        "NotFound",
                        f"The intent {trigger_intent} does not exist in the domain.",
                    )
                await app.agent.trigger_intent(
                    intent_name=intent_to_trigger,
                    entities=entities,
                    output_channel=output_channel,
                    tracker=tracker,
                )
        except ErrorResponse:
            raise
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

        state = tracker.current_state(verbosity)

        response_body: Dict[Text, Any] = {"tracker": state}

        if isinstance(output_channel, CollectingOutputChannel):
            response_body["messages"] = output_channel.messages

        return response.json(response_body)

    @app.post("/conversations/<conversation_id:path>/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    @ensure_conversation_exists()
    async def predict(request: Request, conversation_id: Text) -> HTTPResponse:
        try:
            # Fetches the appropriate bot response in a json format
            responses = await app.agent.predict_next(conversation_id)
            responses["scores"] = sorted(
                responses["scores"], key=lambda k: (-k["score"], k["action"])
            )
            return response.json(responses)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/conversations/<conversation_id:path>/messages")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def add_message(request: Request, conversation_id: Text) -> HTTPResponse:
        validate_request_body(
            request,
            "No message defined in request body. Add a message to the request body in "
            "order to add it to the tracker.",
        )

        request_params = request.json

        message = request_params.get("text")
        sender = request_params.get("sender")
        parse_data = request_params.get("parse_data")

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)

        # TODO: implement for agent / bot
        if sender != "user":
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Currently, only user messages can be passed to this endpoint. "
                "Messages of sender '{}' cannot be handled.".format(sender),
                {"parameter": "sender", "in": "body"},
            )

        user_message = UserMessage(message, None, conversation_id, parse_data)

        try:
            async with app.agent.lock_store.lock(conversation_id):
                tracker = await app.agent.log_message(user_message)

            return response.json(tracker.current_state(verbosity))
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ConversationError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/model/train")
    @requires_auth(app, auth_token)
    @async_if_callback_url
    @run_in_thread
    @inject_temp_dir
    async def train(request: Request, temporary_directory: Path) -> HTTPResponse:
        validate_request_body(
            request,
            "You must provide training data in the request body in order to "
            "train your model.",
        )
        training_payload = _training_payload_from_yaml(request, temporary_directory)

        try:
            from rasa.model_training import train_async

            # pass `None` to run in default executor
            training_result = await train_async(**training_payload)
            user_id = request.args.get("user_id", "user_id")

            if training_result.model:
                model_file = os.path.basename(training_result.model)
                if rasa.utils.endpoints.bool_arg(request, "debug", False):
                    return await response.file(
                        training_result.model,
                        filename=f"{user_id}/" + model_file,
                        headers={"filename": f"{user_id}/" + model_file},
                    )
                return response.json(
                    {
                        "model_file": f"{user_id}/" + model_file,
                    },
                    status=HTTPStatus.OK,
                )

            else:
                raise ErrorResponse(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "TrainingError",
                    "Ran training, but it finished without a trained model.",
                )
        except ErrorResponse as e:
            raise e
        except InvalidDomain as e:
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "InvalidDomainError",
                f"Provided domain file is invalid. Error: {e}",
            )
        except Exception as e:
            logger.error(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TrainingError",
                f"An unexpected error occurred during training. Error: {e}",
            )

    @app.post("/model/test/stories")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app, require_core_is_ready=True)
    @inject_temp_dir
    async def evaluate_stories(
        request: Request, temporary_directory: Path
    ) -> HTTPResponse:
        """Evaluate stories against the currently loaded model."""
        validate_request_body(
            request,
            "You must provide some stories in the request body in order to "
            "evaluate your model.",
        )

        test_data = _test_data_file_from_payload(request, temporary_directory, ".md")

        use_e2e = rasa.utils.endpoints.bool_arg(request, "e2e", default=False)

        try:
            evaluation = await test(
                test_data, app.agent, e2e=use_e2e, disable_plotting=True
            )
            return response.json(evaluation)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TestingError",
                f"An unexpected error occurred during evaluation. Error: {e}",
            )

    @app.post("/model/test/intents")
    @requires_auth(app, auth_token)
    @async_if_callback_url
    @run_in_thread
    @inject_temp_dir
    async def evaluate_intents(
        request: Request, temporary_directory: Path
    ) -> HTTPResponse:
        """Evaluate intents against a Rasa model."""
        validate_request_body(
            request,
            "You must provide some nlu data in the request body in order to "
            "evaluate your model.",
        )

        cross_validation_folds = request.args.get("cross_validation_folds")
        is_yaml_payload = request.headers.get("Content-type") == YAML_CONTENT_TYPE
        test_coroutine = None

        if is_yaml_payload:
            payload = _training_payload_from_yaml(request, temporary_directory)
            config_file = payload.get("config")
            test_data = payload.get("training_files")

            if cross_validation_folds:
                test_coroutine = _cross_validate(
                    test_data, config_file, int(cross_validation_folds)
                )
        else:
            test_data = _test_data_file_from_payload(request, temporary_directory)
            if cross_validation_folds:
                raise ErrorResponse(
                    HTTPStatus.BAD_REQUEST,
                    "TestingError",
                    "Cross-validation is only supported for YAML data.",
                )

        if not cross_validation_folds:
            test_coroutine = _evaluate_model_using_test_set(
                request.args.get("model"), test_data
            )

        try:
            evaluation = await test_coroutine
            return response.json(evaluation)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "TestingError",
                f"An unexpected error occurred during evaluation. Error: {e}",
            )

    async def _evaluate_model_using_test_set(
        model_path: Text, test_data_file: Text
    ) -> Dict:
        logger.info("Starting model evaluation using test set.")

        eval_agent = app.agent

        if model_path:
            model_server = app.agent.model_server
            if model_server is not None:
                model_server = model_server.copy()
                model_server.url = model_path
                # Set wait time between pulls to `0` so that the agent does not schedule
                # a job to pull the model from the server
                model_server.kwargs["wait_time_between_pulls"] = 0
            eval_agent = await _load_agent(
                model_path, model_server, app.agent.remote_storage
            )

        data_path = os.path.abspath(test_data_file)

        if not eval_agent.model_directory or not os.path.exists(
            eval_agent.model_directory
        ):
            raise ErrorResponse(
                HTTPStatus.CONFLICT, "Conflict", "Loaded model file not found."
            )

        model_directory = eval_agent.model_directory
        _, nlu_model = model.get_model_subdirectories(model_directory)

        if nlu_model is None:
            raise ErrorResponse(
                HTTPStatus.CONFLICT,
                "Conflict",
                "Missing NLU model directory.",
            )

        return await rasa.nlu.test.run_evaluation(
            data_path, nlu_model, disable_plotting=True, report_as_dict=True
        )

    async def _cross_validate(data_file: Text, config_file: Text, folds: int) -> Dict:
        logger.info(f"Starting cross-validation with {folds} folds.")
        importer = TrainingDataImporter.load_from_dict(
            config=None, config_path=config_file, training_data_paths=[data_file]
        )
        config = await importer.get_config()
        nlu_data = await importer.get_nlu_data()

        evaluations = rasa.nlu.test.cross_validate(
            data=nlu_data,
            n_folds=folds,
            nlu_config=config,
            disable_plotting=True,
            errors=True,
            report_as_dict=True,
        )
        evaluation_results = _get_evaluation_results(*evaluations)

        return evaluation_results

    def _get_evaluation_results(
        intent_report: CVEvaluationResult,
        entity_report: CVEvaluationResult,
        response_selector_report: CVEvaluationResult,
    ) -> Dict[Text, Any]:
        eval_name_mapping = {
            "intent_evaluation": intent_report,
            "entity_evaluation": entity_report,
            "response_selection_evaluation": response_selector_report,
        }

        result = defaultdict(dict)
        for evaluation_name, evaluation in eval_name_mapping.items():
            report = evaluation.evaluation.get("report", {})
            averages = report.get("weighted avg", {})
            result[evaluation_name]["report"] = report
            result[evaluation_name]["precision"] = averages.get("precision")
            result[evaluation_name]["f1_score"] = averages.get("1-score")
            result[evaluation_name]["errors"] = evaluation.evaluation.get("errors", [])

        return result

    @app.post("/model/predict")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app, require_core_is_ready=True)
    async def tracker_predict(request: Request) -> HTTPResponse:
        """Given a list of events, predicts the next action."""
        validate_events_in_request_body(request)

        verbosity = event_verbosity_parameter(request, EventVerbosity.AFTER_RESTART)
        request_params = request.json
        try:
            tracker = DialogueStateTracker.from_dict(
                DEFAULT_SENDER_ID, request_params, app.agent.domain.slots
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                f"Supplied events are not valid. {e}",
                {"parameter": "", "in": "body"},
            )

        try:
            result = app.agent.create_processor().predict_next_with_tracker(
                tracker, verbosity
            )

            return response.json(result)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "PredictionError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.post("/model/parse")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def parse(request: Request) -> HTTPResponse:
        validate_request_body(
            request,
            "No text message defined in request_body. Add text message to request body "
            "in order to obtain the intent and extracted entities.",
        )
        emulation_mode = request.args.get("emulation_mode")
        emulator = _create_emulator(emulation_mode)

        try:
            data = emulator.normalise_request_json(request.json)
            try:
                parsed_data = await app.agent.parse_message_using_nlu_interpreter(
                    data.get("text")
                )
            except Exception as e:
                logger.debug(traceback.format_exc())
                raise ErrorResponse(
                    HTTPStatus.BAD_REQUEST,
                    "ParsingError",
                    f"An unexpected error occurred. Error: {e}",
                )
            response_data = emulator.normalise_response_json(parsed_data)

            return response.json(response_data)

        except Exception as e:
            logger.debug(traceback.format_exc())
            raise ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "ParsingError",
                f"An unexpected error occurred. Error: {e}",
            )

    @app.put("/model")
    @requires_auth(app, auth_token)
    async def load_model(request: Request) -> HTTPResponse:
        validate_request_body(request, "No path to model file defined in request_body.")

        root_path = "./models/"
        model_path = root_path + request.json.get("model_file", None)
        model_server = request.json.get("model_server", None)
        remote_storage = request.json.get("remote_storage", None)

        if model_server:
            try:
                model_server = EndpointConfig.from_dict(model_server)
            except TypeError as e:
                logger.debug(traceback.format_exc())
                raise ErrorResponse(
                    HTTPStatus.BAD_REQUEST,
                    "BadRequest",
                    f"Supplied 'model_server' is not valid. Error: {e}",
                    {"parameter": "model_server", "in": "body"},
                )

        app.agent = await _load_agent(
            model_path, model_server, remote_storage, endpoints, app.agent.lock_store
        )

        logger.debug(f"Successfully loaded model '{model_path}'.")

        # return response.json(None, status=HTTPStatus.NO_CONTENT)
        return response.json(
            {
                "version": rasa.__version__,
                "status": "success",
                "message": f"Successfully load model: {model_path}",
                "code": 200,
            },
            status=HTTPStatus.OK,
        )

    @app.delete("/model")
    @requires_auth(app, auth_token)
    async def unload_model(request: Request) -> HTTPResponse:
        model_file = app.agent.model_directory

        app.agent = Agent(lock_store=app.agent.lock_store)

        logger.debug(f"Successfully unloaded model '{model_file}'.")
        return response.json(None, status=HTTPStatus.NO_CONTENT)

    @app.get("/domain")
    @requires_auth(app, auth_token)
    @ensure_loaded_agent(app)
    async def get_domain(request: Request) -> HTTPResponse:
        """Get current domain in yaml or json format."""
        accepts = request.headers.get("Accept", default=JSON_CONTENT_TYPE)
        if accepts.endswith("json"):
            domain = app.agent.domain.as_dict()
            return response.json(domain)
        elif accepts.endswith("yml") or accepts.endswith("yaml"):
            domain_yaml = app.agent.domain.as_yaml()
            return response.text(
                domain_yaml, status=HTTPStatus.OK, content_type=YAML_CONTENT_TYPE
            )
        else:
            raise ErrorResponse(
                HTTPStatus.NOT_ACCEPTABLE,
                "NotAcceptable",
                f"Invalid Accept header. Domain can be "
                f"provided as "
                f'json ("Accept: {JSON_CONTENT_TYPE}") or'
                f'yml ("Accept: {YAML_CONTENT_TYPE}"). '
                f"Make sure you've set the appropriate Accept "
                f"header.",
            )

    return app


def _get_output_channel(
    request: Request, tracker: Optional[DialogueStateTracker]
) -> OutputChannel:
    """Returns the `OutputChannel` which should be used for the bot's responses.
    Args:
        request: HTTP request whose query parameters can specify which `OutputChannel`
                 should be used.
        tracker: Tracker for the conversation. Used to get the latest input channel.
    Returns:
        `OutputChannel` which should be used to return the bot's responses to.
    """
    requested_output_channel = request.args.get(OUTPUT_CHANNEL_QUERY_KEY)

    if (
        requested_output_channel == USE_LATEST_INPUT_CHANNEL_AS_OUTPUT_CHANNEL
        and tracker
    ):
        requested_output_channel = tracker.get_latest_input_channel()

    # Interactive training does not set `input_channels`, hence we have to be cautious
    registered_input_channels = getattr(request.app, "input_channels", None) or []
    matching_channels = [
        channel
        for channel in registered_input_channels
        if channel.name() == requested_output_channel
    ]

    # Check if matching channels can provide a valid output channel,
    # otherwise use `CollectingOutputChannel`
    return reduce(
        lambda output_channel_created_so_far, input_channel: (
            input_channel.get_output_channel() or output_channel_created_so_far
        ),
        matching_channels,
        CollectingOutputChannel(),
    )


async def _load_agent(
    model_path: Optional[Text] = None,
    model_server: Optional[EndpointConfig] = None,
    remote_storage: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
    lock_store: Optional[LockStore] = None,
) -> Agent:
    try:
        tracker_store = None
        generator = None
        action_endpoint = None

        if endpoints:
            broker = await EventBroker.create(endpoints.event_broker)
            tracker_store = TrackerStore.create(
                endpoints.tracker_store, event_broker=broker
            )
            generator = endpoints.nlg
            action_endpoint = endpoints.action
            if not lock_store:
                lock_store = LockStore.create(endpoints.lock_store)

        loaded_agent = await rasa.core.agent.load_agent(
            model_path,
            model_server,
            remote_storage,
            generator=generator,
            tracker_store=tracker_store,
            lock_store=lock_store,
            action_endpoint=action_endpoint,
        )
    except Exception as e:
        logger.debug(traceback.format_exc())
        raise ErrorResponse(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "LoadingError",
            f"An unexpected error occurred. Error: {e}",
        )

    if not loaded_agent:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            f"Agent with name '{model_path}' could not be loaded.",
            {"parameter": "model", "in": "query"},
        )

    return loaded_agent


def _docs(sub_url: Text) -> Text:
    """Create a url to a subpart of the docs."""
    return DOCS_BASE_URL + sub_url


def _test_data_file_from_payload(
    request: Request, temporary_directory: Path, suffix: Text = ".tmp"
) -> Text:
    if request.headers.get("Content-type") == YAML_CONTENT_TYPE:
        return str(
            _training_payload_from_yaml(
                request,
                temporary_directory,
                # test stories have to prefixed with `test_`
                file_name=f"{TEST_STORIES_FILE_PREFIX}data.yml",
            )["training_files"]
        )
    else:
        # MD test stories have to be in the `tests` directory
        test_dir = temporary_directory / DEFAULT_CONVERSATION_TEST_PATH
        test_dir.mkdir()
        test_file = test_dir / f"tests{suffix}"
        test_file.write_bytes(request.body)
        return str(test_file)


def _validate_json_training_payload(rjs: Dict) -> None:
    if "config" not in rjs:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            "The training request is missing the required key `config`.",
            {"parameter": "config", "in": "body"},
        )

    if "nlu" not in rjs and "stories" not in rjs:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            "To train a Rasa model you need to specify at least one type of "
            "training data. Add `nlu` and/or `stories` to the request.",
            {"parameters": ["nlu", "stories"], "in": "body"},
        )

    if "stories" in rjs and "domain" not in rjs:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            "To train a Rasa model with story training data, you also need to "
            "specify the `domain`.",
            {"parameter": "domain", "in": "body"},
        )

    if "force" in rjs or "save_to_default_model_directory" in rjs:
        rasa.shared.utils.io.raise_deprecation_warning(
            "Specifying 'force' and 'save_to_default_model_directory' as part of the "
            "JSON payload is deprecated. Please use the header arguments "
            "'force_training' and 'save_to_default_model_directory'.",
            docs=_docs("/api/http-api"),
        )


def _create_emulator(mode: Optional[Text]) -> NoEmulator:
    """Create emulator for specified mode.
    If no emulator is specified, we will use the Rasa NLU format."""

    if mode is None:
        return NoEmulator()
    elif mode.lower() == "wit":
        from rasa.nlu.emulators.wit import WitEmulator

        return WitEmulator()
    elif mode.lower() == "luis":
        from rasa.nlu.emulators.luis import LUISEmulator

        return LUISEmulator()
    elif mode.lower() == "dialogflow":
        from rasa.nlu.emulators.dialogflow import DialogflowEmulator

        return DialogflowEmulator()
    else:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            "Invalid parameter value for 'emulation_mode'. "
            "Should be one of 'WIT', 'LUIS', 'DIALOGFLOW'.",
            {"parameter": "emulation_mode", "in": "query"},
        )


def _training_payload_from_yaml(
    request: Request, temp_dir: Path, file_name: Text = "data.yml"
) -> Dict[Text, Any]:
    logger.debug("Extracting YAML training data from request body.")

    add_args = _extract_additional_arguments(request)
    logger.debug("arg is {}".format(add_args))
    config = select_config(add_args)

    decoded = request.body.decode(rasa.shared.utils.io.DEFAULT_ENCODING)
    _validate_yaml_training_payload(decoded)
    decoded = NLUHandler(add_args, decoded).run()

    training_data = temp_dir / file_name
    config_data = temp_dir / "config.yml"
    rasa.shared.utils.io.write_text_file(decoded, training_data)
    rasa.shared.utils.io.write_text_file(config, config_data)

    user_id = request.args.get("user_id", "user_id")
    project_id = request.args.get("project_id", "project_id")

    model_output_directory = str(temp_dir)
    if rasa.utils.endpoints.bool_arg(request, "save_to_default_model_directory", True):
        model_output_directory = DEFAULT_MODELS_PATH + f"/{user_id}/"

    return dict(
        domain=str(training_data),
        config=str(config_data),
        training_files=str(temp_dir),
        output=model_output_directory,
        fixed_model_name=project_id,
        force_training=rasa.utils.endpoints.bool_arg(request, "force_training", False),
        core_additional_arguments=_extract_core_additional_arguments(request),
        nlu_additional_arguments=_extract_nlu_additional_arguments(request),
    )


def _validate_yaml_training_payload(yaml_text: Text) -> None:
    try:
        RasaYAMLReader().validate(yaml_text)
    except Exception as e:
        raise ErrorResponse(
            HTTPStatus.BAD_REQUEST,
            "BadRequest",
            f"The request body does not contain valid YAML. Error: {e}",
            help_url=DOCS_URL_TRAINING_DATA,
        )


def _extract_core_additional_arguments(request: Request) -> Dict[Text, Any]:
    return {
        "augmentation_factor": rasa.utils.endpoints.int_arg(
            request, "augmentation", 50
        ),
    }


def _extract_nlu_additional_arguments(request: Request) -> Dict[Text, Any]:
    return {
        "num_threads": rasa.utils.endpoints.int_arg(request, "num_threads", 1),
    }


def _extract_additional_arguments(request: Request) -> Dict[Text, Any]:
    return {
        "language": request.args.get("language", "zh"),
        "salesforce": rasa.utils.endpoints.bool_arg(request, "salesforce", False),
        "fallback": rasa.utils.endpoints.bool_arg(request, "fallback", False),
        "fallback_threshold": rasa.utils.endpoints.float_arg(
            request, "fallback_threshold", 0.5
        ),
        "fallback_msg": request.args.get("fallback_msg", "Sorry I didn't get that."),
        "epochs": rasa.utils.endpoints.int_arg(request, "epochs", 100),
        "debug": rasa.utils.endpoints.bool_arg(request, "debug", False),
    }
