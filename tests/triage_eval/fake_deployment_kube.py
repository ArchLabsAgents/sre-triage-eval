"""In-memory stand-in for the kubectl commands adapters/revision_faults.py runs.

Models only what the revision-preserving injection depends on: a template change bumps
deployment.kubernetes.io/revision and is remembered per revision, a strategy change does not, and
`rollout undo --to-revision=N` re-applies revision N's template as a new revision (what the
deployment controller does). `inert=True` makes every mutation a silent no-op that returns stderr,
the way AIOpsLab's KubeCtl.exec_command reports a failed command; `template_patches_fail=True`
lets strategy patches land and rejects template patches (a half-landed inject).
"""

from __future__ import annotations

import copy
import json
import re
import shlex

REVISION = "deployment.kubernetes.io/revision"


def _merge(target, patch):
    """RFC 7386 merge patch."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    out = dict(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = _merge(out.get(key), value)
    return out


def _json_patch(doc, ops):
    doc = copy.deepcopy(doc)
    for op in ops:
        assert op["op"] == "replace", op
        parts = [int(p) if p.isdigit() else p for p in op["path"].strip("/").split("/")]
        node = doc
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = op["value"]
    return doc


class FakeDeploymentKube:
    def __init__(
        self,
        name: str,
        namespace: str,
        containers: list[dict],
        revision: int = 3,
        inert: bool = False,
        template_patches_fail: bool = False,
    ):
        self.name = name
        self.namespace = namespace
        self.inert = inert
        # Strategy patches land, template patches return stderr: a half-landed inject.
        self.template_patches_fail = template_patches_fail
        self.commands: list[str] = []
        template = {"spec": {"containers": containers}}
        self.history = {revision: copy.deepcopy(template)}
        self.deployment = {
            "kind": "Deployment",
            "metadata": {"name": name, "annotations": {REVISION: str(revision)}},
            "spec": {
                "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 1, "maxUnavailable": 0}},
                "template": template,
            },
        }

    @property
    def revision(self) -> int:
        return int(self.deployment["metadata"]["annotations"][REVISION])

    def _set_template(self, template) -> None:
        if template == self.deployment["spec"]["template"]:
            return
        revision = self.revision + 1
        self.deployment["spec"]["template"] = copy.deepcopy(template)
        self.deployment["metadata"]["annotations"][REVISION] = str(revision)
        self.history[revision] = copy.deepcopy(template)

    def exec_command(self, command: str, input_data=None) -> str:
        self.commands.append(command)
        argv = shlex.split(command)
        if argv[:3] == ["kubectl", "get", "deployment"]:
            assert argv[3] == self.name and argv[argv.index("-n") + 1] == self.namespace, command
            return json.dumps(self.deployment)
        if self.inert:
            return "Error from server (Forbidden): simulated\n"
        if argv[:3] == ["kubectl", "patch", "deployment"]:
            patch_type = next(a for a in argv if a.startswith("--type="))[len("--type="):]
            body = json.loads(argv[argv.index("-p") + 1])
            before = self.deployment
            after = _merge(before, body) if patch_type == "merge" else _json_patch(before, body)
            if self.template_patches_fail and after["spec"]["template"] != before["spec"]["template"]:
                return "Error from server (Invalid): simulated template rejection\n"
            self.deployment["spec"]["strategy"] = after["spec"].get("strategy", {})
            self._set_template(after["spec"]["template"])
            return f"deployment.apps/{self.name} patched\n"
        if argv[:3] == ["kubectl", "rollout", "undo"]:
            to = int(re.search(r"--to-revision=(\d+)", command).group(1))
            self._set_template(self.history[to])
            return f"deployment.apps/{self.name} rolled back\n"
        raise AssertionError(f"unexpected command: {command}")
