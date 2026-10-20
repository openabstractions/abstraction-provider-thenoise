# abstraction-provider-thenoise

*For operators and provider authors.*

TheNoise is a standalone image engine, a program that makes pictures from
text prompts on this machine; this provider connects it to OA's native image
contract, `abstraction.inference/image@1`. The engine binary, its portable environment, and model files are
operator supplied; this repository does not bundle or download them.

**What's new here:** A standalone image engine starts when an authorized
request arrives, from a folder proved against its pinned archive, under an OA
resource reservation.

The adapter supports one configured model, one generated image per request,
and bounded PNG output. It maps supported prompt controls to TheNoise's local
HTTP API. Calls are serialized because TheNoise serves one model and one
generation at a time. Active-call cancellation stops the owned engine process
tree and completes only after the supervisor confirms the tree has exited.

## Platform qualification

Windows x64 only: the one pinned artifact is the `win-x64` build, started
through its `thenoise.bat` launcher. Tests of the launcher command and of
8.3 short names skip off Windows.

The pinned Windows `gfx1151` release was exercised on a Ryzen AI MAX+ 395 with
Radeon 8060S. One 512×512 CPU request through TheNoise's official launcher
returned a valid PNG in 176.019 seconds. That is one observation, not a speed
claim. The bundled hipBLASLt and rocBLAS crash on the first GPU request when the
path of a kernel file they open reaches 260 characters with Windows long paths
off. From an engine folder of 87 characters or fewer the unchanged bundle
generates on the GPU. See the [qualification record](../../research/adoption/headless-strix-provider.md),
the [GPU crash findings](../../research/adoption/thenoise-gpu-crash/FINDINGS.md)
and the [security review](../../research/security-reviews/thenoise-0.9.1-rocm10.1.0-gfx1151-win-x64.md)
with its GPU addendum.

## Engine installation and checks

The provider runs one artifact: `thenoise-0.9.1-rocm10.1.0-gfx1151-win-x64.zip`
with SHA-256 `0a639633d6aff9523190c557f2377ba8ef4afab5b02644b3778082793cbdf737`.
The engine folder defaults to `%LOCALAPPDATA%\OA\engines\thenoise-0.9.1`. Under
`C:\Users` that path is 49 characters plus the profile folder name, which fits
the limit for names of up to 38 characters. The provider extracts only into
folders of 87 characters or fewer; a longer folder is refused with a request
to choose a shorter one with `-engine-dir`.

- `-engine-zip` names a local copy of the ZIP. When no manifest has been
  recorded, the provider checks the ZIP's SHA-256 before reading any entry and
  extracts it into an empty or absent engine folder. It records the path,
  SHA-256 and size of every file in a manifest under `-state`. The manifest's
  own SHA-256 is pinned in the source. An existing folder is kept only when it
  equals the ZIP.
- Installing gives every folder of the engine the shared provider kit's
  read-only ACL (inference CONTRACT.md INF-F3). Each launch proves the engine
  folder against that manifest through `providerhost/enginefiles`: a full hash
  at the first launch of the provider process and after any refusal, and
  otherwise a check of each file's ID, size and times against the record
  `thenoise-0.9.1.files` beside the manifest. The files stay open with read
  sharing only until the engine exits. A changed, missing or extra file or
  folder, a link, or a folder without the read-only ACL refuses the launch.
  An engine folder installed before this check lacks the ACL; delete the
  manifest under `-state` and start with `-engine-zip` to install it again.
- `-device auto`, the default, runs on the GPU when the readiness gate passes
  and on the CPU otherwise. `-device cuda` refuses instead of falling back.
  `-device cpu` never uses the GPU.
- The readiness gate runs before every GPU launch. For each file under
  `Lib\site-packages\_rocm_sdk_libraries\bin\hipblaslt\library\gfx1151` and
  `…\bin\rocblas\library\gfx1151` it counts the engine folder's length, 1,
  the file's path below the engine folder, and the separators the library
  inserts: 1 for hipBLASLt, 2 for rocBLAS. Above 259 with `LongPathsEnabled`
  not 1, the provider sets `HIPBLASLT_TENSILE_LIBPATH` and
  `ROCBLAS_TENSILE_LIBPATH` to the 8.3 names of the two `gfx1151` folders. It
  does so only when each 8.3 name is shorter, passes the same count, and
  resolves through `GetFinalPathNameByHandleW` to that folder under the
  bundle's resolved path with no reparse point on the way. Otherwise it
  refuses: "The TheNoise engine folder is too deep for its GPU libraries; move
  it closer to the drive root." `/health` turns ready before any kernel file
  opens and is no evidence of GPU readiness.
- A GPU generation that exits with 0xC0000005 turns GPU launches off until the
  provider restarts.
- ROCm's comgr and MIOpen caches hold GPU code the engine loads unchecked. The
  provider points `AMD_COMGR_CACHE_DIR`, `MIOPEN_USER_DB_PATH` and
  `MIOPEN_CUSTOM_CACHE_DIR` at folders under `-state\rocm-cache`. Each folder
  has a protected ACL granting only the provider account, SYSTEM and
  Administrators. The provider empties the caches when the bundle, the account
  or an ACL changes, or when an entry has another owner or is a link.
- The engine receives the Windows system variables, a `PATH` of the Windows
  system folders, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_HUB_DISABLE_TELEMETRY=1`, `PYTHONDONTWRITEBYTECODE=1` (the engine
  folders are read-only), the three cache folders and, on the GPU, the
  gate's two variables. Credentials and every other provider variable stay
  behind. `thenoise.bat` sets its own `TORCH_BLAS_PREFER_HIPBLASLT`,
  `TORCH_COMPILE_DISABLE` and `TORCHDYNAMO_DISABLE`; the provider never sets
  `ROCBLAS_USE_HIPBLASLT` or `PYTORCH_TUNABLEOP_ENABLED`.

The E5 FP8 DiT checkpoint is refused by the pinned TheNoise loader because it
contains an unsupported `input_scale` tensor. Do not remove or rewrite the
checkpoint metadata. The tested BF16 DiT is licensed Apache-2.0 by Comfy-Org;
TheNoise and model licenses remain separate from this provider. See
[`CONTRACT.md`](CONTRACT.md) and [`NOTICE`](../../research/org-edits/abstraction-provider-thenoise/NOTICE).

The provider uses TheNoise's official Windows `thenoise.bat` launcher so its
bundled ROCm paths and supported environment are active. It binds to an owned
loopback listener through OA's shared identity-bound HTTP client. The executable
keeps TheNoise stopped until an authorized image operation is accepted; model
loading then appears as coarse OA operation progress. The provider publishes
output into its private store; the OA runtime copies verified bytes into the
application-authorized store before delivering a result.

Build `./cmd/thenoise-provider` from the Go module, then register the executable
as an on-demand native provider with an `image@1` declaration for its exact
configured model. The executable requires a designated OA runtime path,
provider endpoint, private store/record paths, the BF16 DiT, VAE and text
encoder paths, and a positive OA resource reservation. The engine folder and
state folder have defaults. For example, its arguments are:

```text
-endpoint <native-provider-endpoint> -runtime <absolute-oa-runtime-exe>
-store <absolute-private-store-dir> -records <absolute-record-file>
-engine-zip <absolute-pinned-thenoise-zip> -dit <absolute-bf16-dit>
-vae <absolute-vae> -text-encoder <absolute-qwen-text-encoder>
-model flux_klein -port 18118 -resource memory
-amount-bytes <operator-memory-reservation>
```

`-state` defaults to `thenoise-state` beside `-records`.

The default resource endpoint is the local OA resource service; use
`-resource-endpoint` when explicitly configured. The native endpoint and
resource endpoint are OA IPC addresses, while port 18118 is private to the
owned engine. The example is a provider process invocation, not a request to
start TheNoise before authorization.

## Development

Requires Go 1.26 or later. Run the focused adapter tests from the abstractions
workspace:

```text
go test ./openabstractions-flat/abstraction-provider-thenoise/go/...
```

The adapter does not implement edit mode, multi-image output, fine-grained
generation progress, or per-request engine unload. Unsupported features are
rejected before the engine is contacted.

## Status

Development provider for `abstraction.inference/image@1`.
