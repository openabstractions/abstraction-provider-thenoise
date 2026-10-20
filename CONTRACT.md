# TheNoise provider contract

## Status

Development adapter for `abstraction.inference/image@1`. The provider wraps a
separately installed TheNoise HTTP engine; it does not bundle TheNoise or model
weights. Its CLI composes the shared image/content provider host and a lazy,
resource-leased engine process.

Binds: [inference.thrift](https://github.com/openabstractions/abstraction-inference/blob/main/inference.thrift)

The conventions every OA contract carries are stated once in [CONVENTIONS.md](https://github.com/openabstractions/abstractions/blob/main/idl/CONVENTIONS.md); a rule here that applies one cites its `CONV-` id.

## Engine boundary

The adapter admits one generated image for its configured model and maps the
prompt, optional dimensions, and explicitly supported `thenoise/*` controls to
TheNoise's synchronous `POST /text2image` endpoint. It accepts only a bounded,
valid PNG response. Edit, multiple images, unknown controls, and required
controls TheNoise cannot honor are rejected before contacting the engine.

TheNoise serves one selected model and has no per-request job identifier,
fine-grained progress, cancellation, or unload operation. OA reports coarse
waiting/loading/generating phases, not an engine completion estimate. Calls are serialized. Cancellation
of an active call stops the owned engine process tree and is complete only when
the supervisor confirms the tree has exited; cancellation while waiting for
the serialized slot does not affect the active call. Engine HTTP is restricted
to the owned loopback listener through the shared identity-bound client.
Once `/health` answers, the provider kit's loopback check (inference
CONTRACT.md INF-EL2) must find the generation's contained process tree
holding its listener on 127.0.0.1 at the launch port and no socket beyond
loopback; otherwise readiness fails and the tree is stopped. The engine
starts with the kit's clean environment (INF-EL1): the Windows system and
profile variables cmd.exe, CPython and the HIP runtime resolve, computed
from the kernel, the token and the registry, a PATH of the Windows system
folders, TEMP and TMP in `engine-temp` under the provider's state, the
reviewed switches and the private ROCm cache folders. Nothing is inherited
from the provider's environment.

## Qualified Windows behavior

The pinned portable gfx1151 bundle works through its official `thenoise.bat`
launcher on the Ryzen AI MAX+ 395 / Radeon 8060S target. A 512x512 CPU request
returned a valid PNG once. See
[`headless-strix-provider.md`](../../research/adoption/headless-strix-provider.md)
for exact versions, hashes, request settings, and evidence. No performance
claim is made from the single CPU request.

The GPU path runs under conditions 8 to 11 of the security review's GPU
addendum. The provider launches `--device cuda` only after its path gate
passes: every kernel file the bundled hipBLASLt and rocBLAS can open must
yield a path of at most 259 characters as they build it, or Windows long paths
must be on, or verified 8.3 names of the two `gfx1151` folders must bring it
within the limit. A GPU generation that ends in an access violation turns the
GPU off until the provider restarts. The ROCm code-object caches live in
provider-owned folders with a private ACL. The engine tree is proven against
the manifest of the pinned ZIP at every launch and held unchanged while the
engine runs (inference CONTRACT.md INF-F1..INF-F5). The recorded GPU run the
addendum requires before shipping is pending.

The E5 FP8 DiT asset was rejected by the pinned loader because of an
unsupported `input_scale` tensor. Do not remove or rewrite checkpoint metadata
to bypass this check. The tested BF16 DiT is a separate Apache-2.0 asset listed
in the qualification record.
