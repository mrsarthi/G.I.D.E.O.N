"""policy.toml is the auditable source of what each tool may do (PRD §5.6)."""

from __future__ import annotations

import pytest

from gideon.core.config import POLICY_FILE
from gideon.tools.base import Param, ToolSpec
from gideon.tools.policy import CLASSES, PolicyError, apply_policy, load_policy
from gideon.tools.registry import all_tools, get_tool


def test_every_tool_is_in_the_policy_file():
    policy = load_policy(POLICY_FILE)
    missing = {t.name for t in all_tools()} - set(policy)
    assert not missing, f"add these to policy.toml: {sorted(missing)}"


def test_every_tool_has_a_valid_class():
    assert {t.capability for t in all_tools()} <= set(CLASSES)


def test_the_file_decides_the_rules():
    shell = get_tool("run_terminal_command")
    assert (shell.capability, shell.confirm, shell.privileged) == ("execute", True, True)
    assert "telegram" in shell.blocked_channels
    assert get_tool("get_weather").capability == "observe"
    assert get_tool("set_reminder").capability == "mutate"


def test_execute_tools_never_reach_remote_channels():
    for tool in all_tools():
        if tool.capability == "execute":
            assert "telegram" in tool.blocked_channels, tool.name
            assert tool.privileged, tool.name


def test_applying_overrides_code_defaults():
    spec = ToolSpec(name="demo", description="", params=(Param("x", "x"),), fn=lambda x: x)
    apply_policy({"demo": spec}, {"demo": {"class": "execute", "confirm": True, "blocked_channels": ["telegram"]}})
    assert (spec.capability, spec.confirm, spec.privileged) == ("execute", True, False)
    assert spec.blocked_channels == frozenset({"telegram"})


@pytest.mark.parametrize(
    "body",
    ['[tools.x]\nclass = "dangerous"\n', '[tools.x]\nclass = "observe"\nconfrim = true\n'],
)
def test_malformed_policy_refuses_to_load(tmp_path, body):
    """A typo in a security file must stop the engine, not silently relax it."""
    path = tmp_path / "policy.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy(path)
