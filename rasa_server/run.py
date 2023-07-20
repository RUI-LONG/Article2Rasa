import os
import uuid
import asyncio
import logging
import shutil
import argparse
import ruamel.yaml
from sanic import Sanic
from ruamel.yaml import YAML
from functools import partial
from typing import Any, List, Optional, Text, Union, Dict

from rasa import server, telemetry
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.utils.io import read_yaml_file, write_yaml
import rasa.utils
import rasa.utils.common
import rasa.utils.io
import rasa.core.utils
from rasa.core import channels, constants
from rasa.core.channels import console
from rasa.core.channels.channel import InputChannel
import rasa.core.interpreter
from rasa.core.utils import AvailableEndpoints
from rasa.core.run import load_agent_on_start, close_resources, create_http_input_channels
from rasa.constants import ENV_SANIC_BACKLOG

from train_component import create_main_app
"""
This File is modified from the original file in rasa.core.run.
https://github.com/RasaHQ/rasa/blob/78a06f3edc90f6b9ae0b21d1e2c4c2fd750dfb70/rasa/core/run.py
"""
logger = logging.getLogger()  # get the root logger

def configure_app(
    input_channels: Optional[List["InputChannel"]] = None,
    cors: Optional[Union[Text, List[Text], None]] = None,
    auth_token: Optional[Text] = None,
    enable_api: bool = True,
    response_timeout: int = constants.DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    route: Optional[Text] = "/webhooks/",
    port: int = constants.DEFAULT_SERVER_PORT,
    endpoints: Optional[AvailableEndpoints] = None,
    log_file: Optional[Text] = None,
    conversation_id: Optional[Text] = uuid.uuid4().hex,
) -> Sanic:
    """Run the agent."""

    rasa.core.utils.configure_file_logging(logger, log_file)

    if enable_api:
        app = create_main_app(
            cors_origins=cors,
            auth_token=auth_token,
            response_timeout=response_timeout,
            jwt_secret=jwt_secret,
            jwt_method=jwt_method,
            endpoints=endpoints,
        )
    else:
        app = _create_app_without_api(cors)

    if input_channels:
        channels.channel.register(input_channels, app, route=route)
    else:
        input_channels = []

    if logger.isEnabledFor(logging.DEBUG):
        rasa.core.utils.list_routes(app)

    async def configure_async_logging() -> None:
        if logger.isEnabledFor(logging.DEBUG):
            rasa.utils.io.enable_async_loop_debugging(asyncio.get_event_loop())

    app.add_task(configure_async_logging)

    if "cmdline" in {c.name() for c in input_channels}:

        async def run_cmdline_io(running_app: Sanic) -> None:
            """Small wrapper to shut down the server once cmd io is done."""
            await asyncio.sleep(1)  # allow server to start

            await console.record_messages(
                server_url=constants.DEFAULT_SERVER_FORMAT.format("http", port),
                sender_id=conversation_id,
            )

            logger.info("Killing Sanic server now.")
            running_app.stop()  # kill the sanic server

        app.add_task(run_cmdline_io)

    return app

def _create_app_without_api(cors: Optional[Union[Text, List[Text]]] = None) -> Sanic:
    app = Sanic(__name__, configure_logging=False)
    server.add_root_route(app)
    server.configure_cors(app, cors)
    return app

def create_endpoint_yml(path, port):
    try:
        read_yaml_file(path)
    except:
        write_yaml({'action_endpoint': {'url': f'http://127.0.0.1:{port}/webhook'}}, path)

def serve_application(
    model_path: Optional[Text] = None,
    channel: Optional[Text] = None,
    port: int = constants.DEFAULT_SERVER_PORT,
    credentials: Optional[Text] = None,
    cors: Optional[Union[Text, List[Text]]] = None,
    auth_token: Optional[Text] = None,
    enable_api: bool = True,
    response_timeout: int = constants.DEFAULT_RESPONSE_TIMEOUT,
    jwt_secret: Optional[Text] = None,
    jwt_method: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
    remote_storage: Optional[Text] = None,
    log_file: Optional[Text] = None,
    ssl_certificate: Optional[Text] = None,
    ssl_keyfile: Optional[Text] = None,
    ssl_ca_file: Optional[Text] = None,
    ssl_password: Optional[Text] = None,
    conversation_id: Optional[Text] = uuid.uuid4().hex,
) -> None:
    """Run the API entrypoint."""

    if not channel and not credentials:
        channel = "cmdline"

    input_channels = create_http_input_channels(channel, credentials)

    app = configure_app(
        input_channels,
        cors,
        auth_token,
        enable_api,
        response_timeout,
        jwt_secret,
        jwt_method,
        port=port,
        endpoints=endpoints,
        log_file=log_file,
        conversation_id=conversation_id,
    )

    ssl_context = server.create_ssl_context(
        ssl_certificate, ssl_keyfile, ssl_ca_file, ssl_password
    )
    protocol = "https" if ssl_context else "http"

    logger.info(
        f"Starting Rasa server on "
        f"{constants.DEFAULT_SERVER_FORMAT.format(protocol, port)}"
    )

    app.register_listener(
        partial(load_agent_on_start, model_path, endpoints, remote_storage),
        "before_server_start",
    )
    app.register_listener(close_resources, "after_server_stop")

    # noinspection PyUnresolvedReferences
    async def clear_model_files(_app: Sanic, _loop: Text) -> None:
        if app.agent.model_directory:
            shutil.rmtree(_app.agent.model_directory)

    number_of_workers = rasa.core.utils.number_of_sanic_workers(
        endpoints.lock_store if endpoints else None
    )

    telemetry.track_server_start(
        input_channels, endpoints, model_path, number_of_workers, enable_api
    )

    app.register_listener(clear_model_files, "after_server_stop")

    # rasa.utils.common.update_sanic_log_level(log_file)
    app.run(
        host="0.0.0.0",
        port=port,
        ssl=ssl_context,
        backlog=int(os.environ.get(ENV_SANIC_BACKLOG, "100")),
        workers=number_of_workers,
    )


def run_service():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--debug', type=bool, required=False, default=False)
    args = parser.parse_args()

    root_dir = ""
    credentials_path = root_dir + "credentials.yml"
    model_path = root_dir + "models"
    endpoints_path = root_dir + f"endpoints/endpoints_{args.port}.yml"

    create_endpoint_yml(endpoints_path, args.port)

    serve_application(
        model_path = model_path,
        endpoints= AvailableEndpoints.read_endpoints(endpoints_path),
        port=args.port,
        credentials =credentials_path,
        cors="*",
        enable_api=True,
    )

if __name__ == '__main__':
    run_service()