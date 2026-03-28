"""Root conftest — create openenv stubs before any package imports."""

import sys
import os
import types
from typing import Optional

if "openenv" not in sys.modules:
    from pydantic import BaseModel, ConfigDict

    class _Action(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: dict = {}

    class _Observation(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = False
        reward: Optional[float] = None
        metadata: dict = {}

    class _State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0

    class _StepResult:
        def __init__(self, observation=None, reward=None, done=True):
            self.observation = observation
            self.reward = reward
            self.done = done

    class _EnvClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __class_getitem__(cls, params): return cls

    class _Environment:
        SUPPORTS_CONCURRENT_SESSIONS = False
        def __init__(self): pass
        def reset(self, **kwargs): pass
        def step(self, action, **kwargs): pass

    class _EnvironmentMetadata(BaseModel):
        name: str = ""
        description: str = ""
        version: str = ""
        author: str = ""

    def _create_app(*args, **kwargs):
        from fastapi import FastAPI
        return FastAPI()

    openenv = types.ModuleType("openenv")
    core = types.ModuleType("openenv.core")
    env_server = types.ModuleType("openenv.core.env_server")
    type_mod = types.ModuleType("openenv.core.env_server.types")
    iface_mod = types.ModuleType("openenv.core.env_server.interfaces")
    http_mod = types.ModuleType("openenv.core.env_server.http_server")
    client_types = types.ModuleType("openenv.core.client_types")

    type_mod.Action = _Action
    type_mod.Observation = _Observation
    type_mod.State = _State
    type_mod.EnvironmentMetadata = _EnvironmentMetadata
    iface_mod.Environment = _Environment
    http_mod.create_app = _create_app
    core.EnvClient = _EnvClient
    client_types.StepResult = _StepResult

    openenv.core = core
    core.env_server = env_server
    env_server.types = type_mod
    env_server.interfaces = iface_mod
    env_server.http_server = http_mod

    sys.modules["openenv"] = openenv
    sys.modules["openenv.core"] = core
    sys.modules["openenv.core.env_server"] = env_server
    sys.modules["openenv.core.env_server.types"] = type_mod
    sys.modules["openenv.core.env_server.interfaces"] = iface_mod
    sys.modules["openenv.core.env_server.http_server"] = http_mod
    sys.modules["openenv.core.client_types"] = client_types
