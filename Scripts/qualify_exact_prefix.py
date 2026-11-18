#!/usr/bin/env python3
"""Qualify the retained CLI runner using public synthetic fixtures and redacted metrics."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import tempfile
import time
from contextlib import closing
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / ".build/release/QwenFieldfareServer"
CORPUS = ROOT / "docs/evidence/2026-11-18-s300/reconstructed-tool-requests.json"


def source(count):
    return "\n\n".join(
        f"def offset_{i}(value: int) -> int:\n    return value + {i}  # deterministic offset"
        for i in range(count)
    )


class Qualification:
    def __init__(self, options, directory):
        self.options = options
        self.secret = secrets.token_hex(32)
        self.key = Path(directory) / "key"
        self.key.touch(mode=0o600)
        self.key.write_text(self.secret + "\n")
        self.report = {"measurements": [], "lifecycle": {}, "context": 8192, "memory_gib": 5,
                       "checkpoint_integrity": "standard model loader checks; reused checkpoint identity",
                       "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest()}
        self.corpus = [x["request"] for x in json.loads(CORPUS.read_text()) if x["label"].startswith("tool-")]
        self.proc = None
        self.log = None
        self.save()

    def save(self):
        self.options.out.mkdir(parents=True, exist_ok=True)
        target = self.options.out / "http-qualification.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)

    def wait(self, predicate, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("Server exited before completing the requested operation")
            try:
                value = self.health()
            except OSError:
                time.sleep(.05)
                continue
            if predicate(value):
                return value
            time.sleep(.025)
        raise RuntimeError("Server did not reach expected state")

    def start(self, mode, label):
        assert self.proc is None
        if subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip() != "1":
            raise RuntimeError("Expected normal memory pressure")
        self.log = (self.options.out / f"server-{label}.log").open("w")
        environment = os.environ.copy()
        environment.pop("QWENFIELDFARE_API_KEY", None)
        command = [str(BIN), "--model", environment["QWENFIELDFARE_SERVER_CHECKPOINT"],
                   "--port", str(self.options.port), "--context", "8192", "--memory-gib", "5",
                   "--queue-capacity", "2", "--api-key-file", str(self.key)]
        if mode is not None:
            command += ["--prefix-cache", mode]
        start = time.monotonic()
        self.proc = subprocess.Popen(command, env=environment, stdout=self.log, stderr=self.log)
        health = self.wait(lambda health: health["status"] == "ready", timeout=120)
        self.report.setdefault("servers", []).append({
            "phase": label, "ready_seconds": time.monotonic() - start, "initial": health,
            "swap_before_requests": self.swap()
        })
        self.save()
        return health

    def stop(self):
        if self.proc is None:
            return
        start = time.monotonic()
        try:
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGTERM)
            code = self.proc.wait(timeout=30)
            self.report["servers"][-1].update(
                exit=code, shutdown_seconds=time.monotonic() - start, swap_after_requests=self.swap()
            )
            assert code == 0, f"Unexpected server exit {code}"
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            self.log.close()
            self.save()

    @staticmethod
    def swap():
        return subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level", "vm.swapusage"], text=True
        ).splitlines()

    def connect(self, path, payload=None, authorized=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.options.port, timeout=450)
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = f"Bearer {self.secret}"
        connection.request("GET" if payload is None else "POST", path,
                           body=None if payload is None else json.dumps(payload), headers=headers)
        return connection

    def call(self, path, payload=None, authorized=True):
        with closing(self.connect(path, payload, authorized)) as connection:
            response = connection.getresponse()
            return response.status, json.loads(response.read())

    def health(self):
        status, health = self.call("/health")
        assert status == 200
        return health

    def clear(self):
        assert self.call("/v1/cache/clear", {}) == (200, {"cleared": True})
        state = self.health()["diagnostics"]["prefixCache"]
        assert not state["valid"] and state["reason"] == "explicitClear"
        return state

    def queued_request(self, payload):
        return self.connect("/v1/chat/completions", payload)

    def wait_idle(self):
        return self.wait(lambda state: not (state["active_generations"] or state["queued_requests"]))

    @staticmethod
    def parse_sse(response):
        output, calls, finish, usage = [], {}, None, None
        for line in response:
            if line == b"data: [DONE]\n":
                assert finish and usage
                message = {"role": "assistant", "content": "".join(output)}
                if calls:
                    message["tool_calls"] = [calls[i] for i in sorted(calls)]
                return message, finish, usage
            if not line.strip():
                continue
            assert line.startswith(b"data: ")
            event = json.loads(line[6:])
            assert "error" not in event
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice["delta"]
                output.append(delta.get("content") or "")
                for part in delta.get("tool_calls", []):
                    call = calls.setdefault(part["index"], {"type": "function", "function": {"arguments": ""}})
                    if "id" in part:
                        call["id"] = part["id"]
                    function = part["function"]
                    if "name" in function:
                        call["function"]["name"] = function["name"]
                    call["function"]["arguments"] += function.get("arguments", "")
        raise AssertionError("Missing SSE terminal marker")

    def measure(self, label, payload, expect_cache=None):
        if payload.get("stream"):
            payload = dict(payload, stream_options={"include_usage": True})
        previous = self.health()["requests"]["completed"]
        start = time.monotonic()
        with closing(self.queued_request(payload)) as connection:
            response = connection.getresponse()
            assert response.status == 200, f"{label}: HTTP {response.status}"
            if payload.get("stream"):
                assert response.getheader("Content-Type") == "text/event-stream"
                message, finish, usage = self.parse_sse(response)
                response.read()
            else:
                value = json.loads(response.read())
                message, finish = (value["choices"][0][key] for key in ("message", "finish_reason"))
                usage = value["usage"]
        elapsed = time.monotonic() - start
        state = self.wait(lambda health: health["requests"]["completed"] == previous + 1)
        timing = state["requests"]["lastRequest"]
        reused = timing["prefixCache"]["reusedPrefixTokens"]
        assert reused + timing["actualPrefillTokens"] == usage["prompt_tokens"]
        if expect_cache is not None:
            assert (reused > 0) == expect_cache, f"{label}: reused {reused} tokens"
        decode = timing.get("decodeAfterFirstTokenMs")
        count = usage["completion_tokens"]
        self.report["measurements"].append({
            "label": label, "stream": payload.get("stream", False), "usage": usage, "finish": finish,
            "http_seconds": elapsed, "timing": timing, "state": state["diagnostics"],
            "decode_tokens_per_second_after_first": (count - 1) * 1000 / decode if count > 1 and decode else None,
            "swap": self.swap()
        })
        self.save()
        print(label, usage, "reused", reused, "seconds", round(elapsed, 3), flush=True)
        return message, finish, usage

    def payload(self, step, history):
        body = deepcopy(self.corpus[step])
        body["messages"] = deepcopy(history)
        return body

    @staticmethod
    def check_tool(completion, index):
        message, finish, _ = completion
        assert finish == "tool_calls"
        calls = message["tool_calls"]
        assert len(calls) == 1 and calls[0]["function"]["name"] == "audit_chunk"
        assert json.loads(calls[0]["function"]["arguments"]) == {"index": index}
        return message

    @staticmethod
    def append_tool(history, message, index, content=None):
        if content is None:
            content = json.dumps({"index": index, "status": "reviewed",
                                  "next_index": index + 1 if index < 3 else None, "source": source(2)})
        history.extend([message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": content}])

    def lifecycle(self):
        case = self.report["lifecycle"]
        ready = self.start("exact-prefix", "opt-in")
        session = ready["diagnostics"]["prefixCache"]["session"]
        case["starts_empty"] = not ready["diagnostics"]["prefixCache"]["valid"]
        assert case["starts_empty"]
        assert self.call("/v1/cache/clear", {}, authorized=False)[0] == 401
        assert self.call("/v1/cache/clear", {"unexpected": 1})[0] == 400
        initial = deepcopy(self.corpus[0]["messages"][:2])
        first = self.check_tool(self.measure("http-tool-json", self.payload(0, initial), False), 1)
        assert self.health()["diagnostics"]["prefixCache"]["valid"]
        continuation = deepcopy(initial)
        self.append_tool(continuation, first, 1)
        second = self.check_tool(self.measure("http-tool-sse", dict(self.payload(1, continuation), stream=True), True), 2)
        assert self.health()["diagnostics"]["prefixCache"]["valid"]
        self.append_tool(continuation, second, 2)
        changed = deepcopy(continuation)
        changed[3]["content"] += " Verified via the alternate reviewer."
        mismatch = self.payload(2, changed)
        output = self.measure("changed-earlier-tool-result", mismatch, False)
        self.check_tool(output, 3)
        assert self.report["measurements"][-1]["timing"]["prefixCache"]["reason"] == "prefixMismatch"
        self.clear()
        replay = self.measure("changed-result-forced-replay", mismatch, False)
        self.check_tool(replay, 3)
        assert output[0]["content"] == replay[0]["content"] and output[1:] == replay[1:]
        case["json_sse_early_result_mismatch_and_replay"] = True
        plain = {"model": "qwen-fieldfare", "temperature": 0, "max_tokens": 16,
                 "messages": [{"role": "user", "content": "Reply with exactly: hello"}]}
        over = dict(plain, messages=[{"role": "user", "content": " repeat" * 8300}])
        assert self.call("/v1/chat/completions", over)[0] == 400
        self.wait_idle()
        state = self.health()["diagnostics"]["prefixCache"]
        assert not state["valid"] and state["reason"] == "rejectedRequest"
        case["context_rejection_invalidates"] = True

        # Hold the active slot during prefill long enough to observe clear and
        # two queued requests. Cancel the second queued client before admission,
        # leaving the first queued request to publish only its own EOS boundary.
        active = self.queued_request(self.payload(0, initial))
        self.wait(lambda state: state["active_generations"] == 1)
        queued = self.queued_request(plain)
        self.wait(lambda state: state["queued_requests"] == 1)
        abandoned = self.queued_request(plain)
        self.wait(lambda state: state["queued_requests"] == 2)
        self.clear()
        abandoned.close()
        with closing(active):
            first_response = active.getresponse()
            assert first_response.status == 200
            first_response.read()
        with closing(queued):
            queued_response = queued.getresponse()
            assert queued_response.status == 200
            result = json.loads(queued_response.read())
            assert "hello" in result["choices"][0]["message"]["content"].lower()
        health = self.wait_idle()
        assert health["requests"]["cancelled"] >= 1 and not health["diagnostics"]["prefixCache"]["valid"]
        case["clear_queue_order_and_queued_disconnect"] = True

        with closing(self.queued_request(dict(self.payload(0, initial), stream=True))):
            self.wait(lambda state: state["active_generations"] == 1)
        self.wait_idle()
        assert self.health()["requests"]["cancelled"] >= 2
        case["active_disconnect"] = True
        pending = self.queued_request(self.payload(0, initial))
        with closing(pending):
            self.wait(lambda state: state["active_generations"] == 1)
            self.stop()
        restart = self.start(None, "default-restart")
        new = restart["diagnostics"]["prefixCache"]
        assert new["session"] != session and not new["valid"] and new["mode"] == "off"
        case["active_shutdown_restart_default_off"] = True
        self.measure("restarted-default-off", plain, False)
        self.stop()

    def performance(self):
        self.start("exact-prefix", "performance")
        initial = self.corpus[0]["messages"][:2]
        for label, size, lower, upper in [("1k", 32, 950, 1600), ("4k", 143, 3750, 4650), ("long", 217, 5650, 7250)]:
            self.clear()
            history = deepcopy(initial)
            if size != 32:
                history[-1]["content"] = history[-1]["content"].split("\nModule:\n", 1)[0] + "\nModule:\n" + source(size)
            setup = self.measure(f"{label}-setup", self.payload(0, history), False)
            self.append_tool(history, self.check_tool(setup, 1), 1)
            payload = self.payload(1, history)
            reused = self.measure(f"{label}-reuse", payload, True)
            assert lower <= reused[2]["prompt_tokens"] <= upper
            self.clear()
            replay = self.measure(f"{label}-full-replay", payload, False)
            self.check_tool(reused, 2)
            self.check_tool(replay, 2)
            assert reused[0]["content"] == replay[0]["content"] and reused[1:] == replay[1:]
        self.stop()

    def opencode(self, directory, mode="exact-prefix"):
        self.start(mode, "opencode" if mode == "exact-prefix" else "opencode-replay")
        work = Path(directory) / "work"
        work.mkdir()
        subprocess.run(["git", "init", "-q", str(work)], check=True)
        implementation = work / "calculator.py"
        implementation.write_text("def add(a, b):\n    return a - b\n")
        (work / "test_calculator.py").write_text(
            "import unittest\nfrom calculator import add\n\n"
            "class AdditionTest(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"
        )
        config = {
            "model": "fieldfare/qwen-fieldfare", "small_model": "fieldfare/qwen-fieldfare",
            "provider": {"fieldfare": {
                "npm": "@ai-sdk/openai-compatible", "name": "QwenFieldfare",
                "options": {"baseURL": f"http://127.0.0.1:{self.options.port}/v1",
                            "apiKey": "{env:QWENFIELDFARE_API_KEY}", "timeout": 600000},
                "models": {"qwen-fieldfare": {"name": "QwenFieldfare local", "tool_call": True,
                                            "temperature": True, "limit": {"context": 8192, "output": 512}}}
            }},
            "agent": {"fieldfare": {
                "description": "Disposable local qualification", "mode": "primary",
                "prompt": "Use allowed tools one at a time. Work only in the disposable workspace. "
                          "Read calculator.py directly, edit it, and run the one allowed test command.",
                "temperature": 0, "steps": 8,
                "permission": {"*": "deny", "read": "allow", "edit": {"*": "deny", "calculator.py": "allow"},
                               "bash": {"*": "deny", "python3 -m unittest -v": "allow"},
                               "external_directory": "deny"}
            }},
            "autoupdate": False, "share": "disabled"
        }
        (work / "opencode.json").write_text(json.dumps(config))
        env = dict(os.environ, PWD=str(work), QWENFIELDFARE_API_KEY=self.secret,
                   OPENCODE_DISABLE_MODELS_FETCH="1", OPENCODE_DISABLE_AUTOUPDATE="1",
                   OPENCODE_DISABLE_CLAUDE_CODE="1", OPENCODE_DISABLE_EXTERNAL_SKILLS="1",
                   OPENCODE_DISABLE_AUTOCOMPACT="1", OPENCODE_DISABLE_LSP_DOWNLOAD="1")
        for kind in ("CACHE", "CONFIG", "DATA", "STATE"):
            env[f"XDG_{kind}_HOME"] = str(Path(directory) / kind.lower())
        version = subprocess.check_output(["opencode", "--version"], env=env, text=True).strip()
        prompt = ("Read calculator.py using the read tool. Make add(a,b) return a+b using the edit tool. "
                  "Then run exactly python3 -m unittest -v using the bash tool. "
                  "After the test passes, report the outcome in one sentence. Do not use any other tools.")
        command = ["opencode", "run", "--pure", "--format", "json", "--dir", str(work),
                   "--model", "fieldfare/qwen-fieldfare", "--agent", "fieldfare", "--title",
                   "QwenFieldfare prefix qualification", prompt]
        start = time.monotonic()
        requests = []
        with tempfile.TemporaryFile(mode="w+") as events:
            client = subprocess.Popen(command, cwd=work, env=env, stdout=events, stderr=subprocess.DEVNULL)
            try:
                self.report["opencode_progress"] = {"version": version, "mode": mode, "requests": requests}
                self.save()
                completed = 0
                while client.poll() is None and time.monotonic() - start < 1200:
                    health = self.health()
                    found = health["requests"]["completed"]
                    if found > completed:
                        requests.append({"completion": found, "timing": health["requests"]["lastRequest"],
                                         "diagnostics": health["diagnostics"]})
                        completed = found
                        self.save()
                    time.sleep(.15)
                if client.poll() is None:
                    raise RuntimeError("OpenCode workflow exceeded the 20-minute bound")
                health = self.wait_idle()
                found = health["requests"]["completed"]
                if found > completed:
                    requests.append({"completion": found, "timing": health["requests"]["lastRequest"],
                                     "diagnostics": health["diagnostics"]})
                events.seek(0)
                data = [json.loads(line) for line in events if line.startswith("{")]
            finally:
                if client.poll() is None:
                    client.terminate()
                    try:
                        client.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        client.kill()
                        client.wait()
        tools = [event["part"] for event in data if event.get("type") == "tool_use"]
        observed = [tool["tool"] for tool in tools if tool["state"]["status"] == "completed"]
        invoked_test = any(tool["tool"] == "bash" and "OK" in tool["state"].get("output", "") for tool in tools)
        final = any(event.get("type") == "text" and event.get("part", {}).get("text", "").strip() for event in data)
        model_edited = implementation.read_text() != "def add(a, b):\n    return a - b\n"
        tests = subprocess.run(["python3", "-m", "unittest", "-v"], cwd=work, capture_output=True, timeout=20)
        del self.report["opencode_progress"]
        self.report["opencode"] = {
            "mode": mode, "version": version, "client_seconds": time.monotonic() - start, "exit": client.returncode,
            "completed_tools": observed, "test_seen_in_tool_result": invoked_test, "workspace_edit": model_edited,
            "independent_tests_exit": tests.returncode, "final_response": final, "requests": requests,
            "reused_token_counts": [row["timing"]["prefixCache"]["reusedPrefixTokens"] for row in requests]
        }
        self.save()
        assert (client.returncode == 0 and tests.returncode == 0 and model_edited and invoked_test and final
                and set(observed) >= {"read", "edit", "bash"}), "OpenCode workflow did not pass"
        self.stop()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--port", type=int, default=18096)
    cli.add_argument("--phase", choices=("lifecycle", "performance", "opencode", "opencode-replay", "all"), default="all")
    options = cli.parse_args()
    with tempfile.TemporaryDirectory(prefix="qwen-prefix-qualification-") as directory:
        trial = Qualification(options, directory)
        try:
            if options.phase in ("lifecycle", "all"):
                trial.lifecycle()
            if options.phase in ("performance", "all"):
                trial.performance()
            if options.phase in ("opencode", "all"):
                trial.opencode(directory)
            if options.phase == "opencode-replay":
                trial.opencode(directory, mode="off")
            trial.report["passed"] = True
        finally:
            trial.stop()
            trial.save()


if __name__ == "__main__":
    main()
