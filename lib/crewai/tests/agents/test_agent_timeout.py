"""Regression tests for ``Agent.max_execution_time`` enforcement.

A timed-out execution has to release the caller when the timeout fires. If
the wrapper instead waits for the timed-out worker to finish before raising,
``max_execution_time`` protects nothing: a hung LLM call or blocked tool
keeps the task parked for as long as the hang lasts, which is the class of
production failure the parameter exists to bound.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from crewai import Agent, Task


def _make_agent(**overrides: Any) -> Agent:
    return Agent(
        role="Timeout Tester",
        goal="Exercise max_execution_time enforcement.",
        backstory="Regression tests for timeout handling.",
        allow_delegation=False,
        **overrides,
    )


def _make_task(agent: Agent) -> Task:
    return Task(
        description="Wait for the timeout.",
        expected_output="Never reached.",
        agent=agent,
    )


def test_execution_within_the_limit_returns_its_result():
    """A fast execution returns its result unchanged."""
    agent = _make_agent()
    task = _make_task(agent)

    with patch.object(Agent, "_execute_without_timeout", return_value="ok"):
        assert agent._execute_with_timeout("prompt", task, 30) == "ok"


def test_timed_out_execution_releases_the_caller_at_the_timeout():
    """The TimeoutError must surface when the timeout expires, not after the hang."""
    agent = _make_agent()
    task = _make_task(agent)
    release = threading.Event()

    def hang(*args: Any, **kwargs: Any) -> str:
        # Stands in for a hung LLM call or blocked tool: nothing inside the
        # worker stops on its own, like the production hangs
        # max_execution_time exists to bound.
        release.wait(timeout=15)
        return "too late"

    started = time.monotonic()
    try:
        with patch.object(Agent, "_execute_without_timeout", hang):
            with pytest.raises(TimeoutError, match="timed out after 1 seconds"):
                agent._execute_with_timeout("prompt", task, 1)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 5, (
        "the timeout must release the caller when it fires; waiting for the "
        f"hung worker before raising took {elapsed:.1f}s"
    )


def test_execute_task_enforces_the_limit_on_a_hung_execution():
    """The public entry point bounds a hung execution too."""
    agent = _make_agent(max_execution_time=1)
    task = _make_task(agent)
    release = threading.Event()

    def hang(*args: Any, **kwargs: Any) -> str:
        release.wait(timeout=15)
        return "too late"

    started = time.monotonic()
    try:
        with patch.object(Agent, "_execute_without_timeout", hang):
            with pytest.raises(TimeoutError):
                agent.execute_task(task)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 5, f"execute_task blocked {elapsed:.1f}s past the timeout"
