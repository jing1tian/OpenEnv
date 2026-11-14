# SPDX-License-Identifier: BSD-3-Clause

"""An idle socket must lose access before its environment is released."""

import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, ConcurrencyConfig, Observation, State
from starlette.websockets import WebSocketDisconnect


class IdleEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self.closed = threading.Event()
        self.close_count = 0

    def reset(self, **kwargs):
        return Observation()

    def step(self, action, **kwargs):
        # Outlast both the idle timeout and a background reaper interval.
        time.sleep(5.2)
        assert not self.closed.is_set()
        return Observation()

    @property
    def state(self):
        return State()

    def close(self):
        self.close_count += 1
        self.closed.set()


@pytest.mark.parametrize("route", ["/ws", "/mcp"])
def test_idle_socket_closes_and_capacity_is_reusable(route):
    server = HTTPEnvServer(
        env=IdleEnvironment,
        action_cls=Action,
        observation_cls=Observation,
        concurrency_config=ConcurrencyConfig(
            max_concurrent_envs=1, session_timeout=0.2
        ),
    )
    app = FastAPI()
    server.register_routes(app)
    message = (
        {"type": "reset", "data": {}}
        if route == "/ws"
        else {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )
    with TestClient(app) as client:
        with client.websocket_connect(route) as ws:
            ws.send_json(message)
            ws.receive_json()
            env = next(iter(server._sessions.values()))
            assert env.closed.wait(2), "Idle session was not closed"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        assert server.active_sessions == 0
        assert env.close_count == 1
        assert not server._session_websocket_attachments
        with client.websocket_connect(route) as replacement:
            replacement.send_json(message)
            replacement.receive_json()
            assert next(iter(server._sessions.values())) is not env


def test_idle_timeout_does_not_expire_an_inflight_action():
    server = HTTPEnvServer(
        env=IdleEnvironment,
        action_cls=Action,
        observation_cls=Observation,
        concurrency_config=ConcurrencyConfig(
            max_concurrent_envs=1, session_timeout=0.2
        ),
    )
    app = FastAPI()
    server.register_routes(app)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {}})
            ws.receive_json()
            env = next(iter(server._sessions.values()))
            ws.send_json({"type": "step", "data": {}})
            assert ws.receive_json()["type"] == "observation"
            assert not env.closed.is_set()
            assert env.closed.wait(2)
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
