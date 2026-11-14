import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.model_ssh import (
    load_config,
    resolve_profile,
    build_ssh_args,
    export_vscode_config,
    DEFAULT_CONFIG_FILE
)


class TestModelSsh(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG_FILE)

    def test_default_config_structure(self):
        self.assertIn("default_profile", self.config)
        self.assertIn("profiles", self.config)
        self.assertIn("port_presets", self.config)

        profiles = self.config["profiles"]
        self.assertIn("neri-cloud", profiles)
        self.assertIn("gpu-training", profiles)

        neri = profiles["neri-cloud"]
        self.assertEqual(neri["host"], "47.108.14.199")
        self.assertEqual(neri["user"], "codex")
        self.assertIn(8002, neri["forward_ports"])

    def test_resolve_profile_default(self):
        profile = resolve_profile(self.config)
        self.assertEqual(profile["name"], "neri-cloud")
        self.assertEqual(profile["host"], "47.108.14.199")
        self.assertEqual(profile["forward_ports"], [8002])

    def test_resolve_profile_overrides(self):
        overrides = {
            "host": "192.168.1.100",
            "port": 2222,
            "user": "developer",
            "forward_ports": [6006, 7860],
            "tmux_session": "yolo-training"
        }
        profile = resolve_profile(self.config, "gpu-training", overrides=overrides)
        self.assertEqual(profile["host"], "192.168.1.100")
        self.assertEqual(profile["port"], 2222)
        self.assertEqual(profile["user"], "developer")
        self.assertEqual(profile["forward_ports"], [6006, 7860])
        self.assertEqual(profile["tmux_session"], "yolo-training")

    def test_resolve_profile_no_forward(self):
        profile = resolve_profile(self.config, "neri-cloud", overrides={"no_forward": True})
        self.assertEqual(profile["forward_ports"], [])

    def test_resolve_profile_invalid_name(self):
        with self.assertRaises(ValueError):
            resolve_profile(self.config, "non-existent-server")

    def test_build_ssh_args_basic(self):
        profile = {
            "name": "test-box",
            "host": "10.0.0.5",
            "port": 22,
            "user": "root",
            "forward_ports": [6006, 8888]
        }
        args = build_ssh_args(profile, forward=True)
        self.assertEqual(args[0], "ssh")
        self.assertIn("ServerAliveInterval=60", args)
        self.assertIn("ServerAliveCountMax=3", args)
        self.assertIn("TCPKeepAlive=yes", args)
        self.assertIn("6006:127.0.0.1:6006", args)
        self.assertIn("8888:127.0.0.1:8888", args)
        self.assertEqual(args[-1], "root@10.0.0.5")

    def test_build_ssh_args_custom_port_and_key(self):
        profile = {
            "name": "custom-box",
            "host": "gpu.autodl.com",
            "port": 34567,
            "user": "root",
            "key_path": "",
            "forward_ports": []
        }
        args = build_ssh_args(profile, forward=False)
        self.assertIn("-p", args)
        port_idx = args.index("-p")
        self.assertEqual(args[port_idx + 1], "34567")
        self.assertEqual(args[-1], "root@gpu.autodl.com")

    def test_build_ssh_args_tmux_command(self):
        profile = {
            "name": "test-tmux",
            "host": "10.0.0.2",
            "user": "ubuntu",
            "forward_ports": []
        }
        tmux_cmd = ["-t", "tmux new-session -A -s my-model"]
        args = build_ssh_args(profile, extra_cmd=tmux_cmd)
        self.assertIn("-t", args)
        self.assertIn("tmux new-session -A -s my-model", args)


if __name__ == "__main__":
    unittest.main()
