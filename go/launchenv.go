package thenoise

import (
	"github.com/openabstractions/abstraction-facade/go/supervision"
	"os"
	"path/filepath"
	"strings"
)

// systemVariables are the Windows system and profile variables cmd.exe,
// CPython and the HIP runtime resolve. The provider kit computes each from
// the kernel, the token or the registry (inference CONTRACT.md INF-EL1);
// none is read from the provider's environment. SystemRoot, windir,
// SystemDrive, PATH, TEMP and TMP are the kit's own. Every credential, proxy,
// PYTHON*, HF_*, TORCH_*, PYTORCH_* and ROCm/HIP variable of the provider
// stays behind.
var systemVariables = []string{
	"ALLUSERSPROFILE", "APPDATA", "COMPUTERNAME", "ComSpec",
	"CommonProgramFiles", "CommonProgramFiles(x86)", "CommonProgramW6432",
	"HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "NUMBER_OF_PROCESSORS", "OS",
	"PATHEXT", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
	"PROCESSOR_LEVEL", "PROCESSOR_REVISION", "ProgramData", "ProgramFiles",
	"ProgramFiles(x86)", "ProgramW6432", "PUBLIC", "USERDOMAIN", "USERNAME",
	"USERPROFILE",
}

// reviewedVariables are condition 3 of research/security-reviews/
// thenoise-0.9.1-rocm10.1.0-gfx1151-win-x64.md. thenoise.bat sets
// TORCH_BLAS_PREFER_HIPBLASLT, TORCH_COMPILE_DISABLE and TORCHDYNAMO_DISABLE
// itself, and condition 11 keeps them as it ships them; the provider sets
// none of the three, nor ROCBLAS_USE_HIPBLASLT or PYTORCH_TUNABLEOP_ENABLED.
// PYTHONDONTWRITEBYTECODE=1 keeps CPython from trying to write compiled
// files into the engine folder, whose folders are read-only (inference
// CONTRACT.md INF-F3); it changes no loaded code, and the review of
// 2026-10-01 admits it.
var reviewedVariables = [...]string{
	"HF_HUB_OFFLINE=1",
	"TRANSFORMERS_OFFLINE=1",
	"HF_HUB_DISABLE_TELEMETRY=1",
	"PYTHONDONTWRITEBYTECODE=1",
}

// engineTemp is the engine's private temporary folder in the provider's
// state: TEMP and TMP name it.
func engineTemp(stateDir string) string { return filepath.Join(stateDir, "engine-temp") }

// launchEnv builds the complete engine environment through the provider
// kit's clean-environment builder: the system variables, a PATH of the
// Windows system folders (thenoise.bat prepends the bundle's own), TEMP and
// TMP in the provider's state, the reviewed variables, the three cache
// folders under stateDir, and gpu, the gate's TENSILE_LIBPATH variables for
// a GPU launch (empty for CPU). A system variable that cannot be computed is
// left out and reported through onError.
func launchEnv(stateDir string, gpu []string, onError func(error)) ([]string, error) {
	temp := engineTemp(stateDir)
	if err := os.MkdirAll(temp, 0o700); err != nil {
		return nil, err
	}
	set := append([]string(nil), reviewedVariables[:]...)
	for _, c := range cacheFolders {
		set = append(set, c.variable+"="+filepath.Join(cacheRoot(stateDir), c.folder))
	}
	for _, kv := range gpu {
		name, _, _ := strings.Cut(kv, "=")
		for _, lib := range gpuLibraries {
			if name == lib.envVar {
				set = append(set, kv)
			}
		}
	}
	return supervision.CleanEnvironment{Temp: temp, WindowsDefaultPath: true, Names: systemVariables, Set: set, OnError: onError}.Build()
}
