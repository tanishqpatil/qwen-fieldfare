# Third-party software and model terms

QwenFieldfare repository source is licensed under the
[Apache License 2.0](LICENSE). That license applies to this repository's source
and documentation. It does not relicense model weights or third-party packages.

This file records the dependency review performed on 2026-07-15. It is an
attribution aid, not legal advice. Anyone distributing a compiled product must
also preserve the license and NOTICE material required by the exact dependency
versions included in that product.

## Upstream project

This derivative contains software based on
[`drumih/turbo-fieldfare`](https://github.com/drumih/turbo-fieldfare), created
by Andrey Mikhaylov and distributed under the Apache License 2.0. The original
license and attribution are preserved in `LICENSE` and `NOTICE`.

## Model weights

Model weights are not included in this source repository. The Qwen runtime is
implemented for the local MLX Q4 checkpoint
[`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit),
which is a quantization of
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
Both model repositories identify their artifacts as Apache-2.0.

Downloaded or separately redistributed weights remain separate works governed
by their source terms. Anyone publishing a model copy or modification should
preserve the applicable license and model-card/source attribution and describe
their changes. The upstream model repositories are authoritative.

## Swift package graph

The following table covers the complete graph reported by
`swift package show-dependencies` from the checked-in
[`Package.resolved`](Package.resolved). Exact revisions are recorded there.

| Package | Version | License in locked checkout |
| --- | --- | --- |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.3.3 | Apache-2.0 |
| [swift-jinja](https://github.com/huggingface/swift-jinja) | 2.3.6 | Apache-2.0 |
| [swift-huggingface](https://github.com/huggingface/swift-huggingface) | 0.9.0 | Apache-2.0 |
| [EventSource](https://github.com/mattt/EventSource) | 1.4.1 | MIT |
| [swift-nio](https://github.com/apple/swift-nio) | 2.101.3 | Apache-2.0; upstream NOTICE applies |
| [swift-atomics](https://github.com/apple/swift-atomics) | 1.3.0 | Apache-2.0 with Runtime Library Exception |
| [swift-collections](https://github.com/apple/swift-collections) | 1.5.1 | Apache-2.0 with Runtime Library Exception |
| [swift-system](https://github.com/apple/swift-system) | 1.6.4 | Apache-2.0 with Runtime Library Exception |
| [swift-crypto](https://github.com/apple/swift-crypto) | 4.5.0 | Apache-2.0; upstream NOTICE applies |
| [swift-asn1](https://github.com/apple/swift-asn1) | 1.7.0 | Apache-2.0; upstream NOTICE applies |
| [yyjson](https://github.com/ibireme/yyjson) | 0.12.0 | MIT |

No copyleft or custom non-commercial license was found in this resolved graph.
The dependency license files remain authoritative. For binary distribution,
collect their license and NOTICE files from the exact revisions in
`Package.resolved`; do not treat this summary as a substitute for that bundle.
