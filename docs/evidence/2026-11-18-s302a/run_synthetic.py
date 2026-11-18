#!/usr/bin/env python3
"""Build the disposable benchmark from the exact archived selector declarations."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_checkout", type=Path)
    args = parser.parse_args()
    source = (args.probe_checkout / "Sources/QwenFieldfareCore/Runtime/Generation"
              / "QwenGenerationSupport.swift").read_text()
    configuration = source[source.index("public enum QwenGenerationConfigurationError"):
                           source.index("public enum QwenSpecialTokens")]
    probe = source[source.index("/// One completed step of the opt-in"):
                   source.index("/// Shared generation state machine.")]
    selector = source[source.index("enum QwenLogitSelector"):]
    harness = Path(__file__).with_name("synthetic.swift").read_text()
    with tempfile.TemporaryDirectory(prefix="qwen-selector-synthetic-") as directory:
        main_swift = Path(directory) / "main.swift"
        main_swift.write_text("\n".join([
            "import Accelerate\nimport Darwin\nimport Foundation", configuration, probe, selector, harness,
        ]))
        binary = Path(directory) / "synthetic"
        subprocess.run(["swiftc", "-O", "-whole-module-optimization", str(main_swift),
                        "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
