# splitwire — SWP/1 engine-free tools

Engine-agnostic pipeline-split inference: the wire protocol (SWP/1), a weight server/puller, a layer planner,
a deterministic toy stage for conformance testing, and a pure-Python client that adapters build on.
The protocol spec and the adapter / engine-change plans live in the accompanying doc.

## Build
Needs Rust >= 1.75. `Cargo.lock` pins dependencies that work on 1.75 (older distro cargo).

    cargo build --release          # binaries in target/release/
    cargo test --release           # framing unit tests
    ./target/release/swp-drive --selftest     # 18 conformance checks (split==unsplit, rollback, auth, ...)

## Tools
| binary | purpose |
|---|---|
| `swp-weightd index` | tensor-level manifest (safetensors or GGUF), blake3 per tensor, layer/group/routed classification |
| `swp-weightd serve` | serve manifest + tensors (chunked, hashed) over SWP; refuses to start without `--secret-env` or `--insecure` |
| `swp-weightd pull`  | pull only a stage's tensors into a normal local safetensors shard (+ config/tokenizer files); engines load it unchanged |
| `swp-plan`          | pick layer ranges per node (tiers: VRAM/RAM/...), bandwidth-bound estimate; emits plan.json that `pull --plan --node` consumes |
| `swp-mockstage`     | deterministic toy stage server (bit-exact across languages) for adapter conformance |
| `swp-drive`         | reference coordinator, `--selftest`, and `--bench` (per-stage time + prefill/decode tok/s) and `--probe` (link RTT/throughput + a verdict on whether boundary quantisation is worth it) |
| `python/bench.py`   | same per-stage timing/tok/s accounting for a Python coordinator (`Trace`) |
| `python/swp.py`     | stdlib Python client (framing, handshake+auth, stage RPCs); `python/test_interop.py` checks it against Rust |

## Typical flow
    # on the machine that holds the weights
    export SWP_SECRET=...; swp-weightd serve --model /models/GLM-5.3-Flash-exl3 --secret-env SWP_SECRET
    swp-weightd index --model /models/GLM-5.3-Flash-exl3 --out manifest.json
    swp-plan --manifest manifest.json --nodes nodes.json --experts 8/288 --replicate-embed-last --out plan.json
    # on the other machine
    swp-weightd pull --server strix:7700 --plan plan.json --node razer --out /models/razer-slice --secret-env SWP_SECRET

## Known limits (v0)
* TENSOR_GET streams whole tensors; byte-range reads (for direct-from-network loading) are specified but not implemented.
* `pull` writes safetensors only; GGUF is index/serve only.
* Authentication is a keyed-hash challenge, not encryption. Use on a direct cable / trusted LAN.
* Thread-per-connection, single-user scale. Planner estimates rank placements; they are not predictions.

## Benchmark: where does the time go, and tokens/s
    swp-mockstage --listen 0.0.0.0:7801 --sim-base-ms 3 --sim-row-us 10     # toy stage with simulated compute
    swp-drive --bench --stages a:7801,b:7802 --bounds 4 --prompt-len 256 --n 64 --chunk 64 --json bench.json
Per stage: compute (the stage's own `stage_ms`), wire+queue (round trip minus compute), coordinator time; for prefill and
decode separately, with tok/s, p50/p95 token latency and time to first token. Real stages must sync their device before
reading their clock, or compute is under-reported and the gap shows up as wire+queue.

## Probe your link (works against any SWP endpoint that echoes PING: mockstage, weightd, real stages)
    swp-drive --probe --stages strix-tb:7801 --hidden 6144 --decode-ms 35 --prefill-tok-s 3000
    swp-drive --probe --stages strix-tb:7801 --assume-mbps 115     # what-if: a 1 GbE link
Reports RTT, decode-shaped echoes per dtype, sustained throughput, and a decode / prefill verdict. Pass your real
`--hidden`; the default is 4096 (GLM-5.3-Flash as reported by a reviewer; check your config.json).
`--assume-rtt-us` and `--assume-mbps` run what-ifs.
