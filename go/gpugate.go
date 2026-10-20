package thenoise

import (
	"errors"
	"github.com/openabstractions/abstraction-facade/go/supervision"
	"path/filepath"

	"github.com/openabstractions/abstraction-provider-kit/go/enginelaunch"
)

// MaxEngineDirLength is the longest folder holding thenoise.bat, in UTF-16
// code units, from which the pinned bundle's GPU libraries open every kernel
// file with LongPathsEnabled off (measured in
// research/adoption/thenoise-gpu-crash/FINDINGS.md). The provider extracts
// only into folders no longer than this.
const MaxEngineDirLength = 87

// maxLibraryPath is the longest path the ROCm libraries open successfully
// without Windows long-path support; the provider kit's gate owns it.
const maxLibraryPath = supervision.MaxLibraryPath

// ErrGPUPathTooDeep is the refusal of a GPU launch whose kernel library files
// the bundled hipBLASLt or rocBLAS could not open. It is one plain sentence
// because the operator reads it.
var ErrGPUPathTooDeep = errors.New("The TheNoise engine folder is too deep for its GPU libraries; move it closer to the drive root.")

// gpuLibrary is one ROCm library of the bundle: its subfolder under bin, the
// variable that points it at its gfx1151 folder, and the separators it adds
// (enginelaunch.GPULibrary).
type gpuLibrary struct {
	name   string
	envVar string
	extra  int
}

var gpuLibraries = [...]gpuLibrary{
	{name: "hipblaslt", envVar: "HIPBLASLT_TENSILE_LIBPATH", extra: 1},
	{name: "rocblas", envVar: "ROCBLAS_TENSILE_LIBPATH", extra: 2},
}

// libraryDir is a library's gfx1151 kernel folder inside the bundle.
func libraryDir(engineDir, library string) string {
	return filepath.Join(engineDir, "Lib", "site-packages", "_rocm_sdk_libraries", "bin", library, "library", "gfx1151")
}

func utf16Len(s string) int { return supervision.UTF16Len(s) }

// gpuProbe holds the operating-system questions the gate asks.
type gpuProbe = enginelaunch.GPUProbe

func systemGPUProbe() gpuProbe { return enginelaunch.SystemGPUProbe() }

// gpuGate is the readiness gate for a --device cuda launch (security review
// condition 8): the provider kit's shared path-length gate on the bundle's
// two gfx1151 folders (inference CONTRACT.md INF-EL3). It returns the extra
// environment the launch needs: nothing when every kernel path fits or
// Windows long paths are on, or both TENSILE_LIBPATH variables set to the
// verified 8.3 names of the two folders (condition 9). It refuses with
// ErrGPUPathTooDeep otherwise. /health turns ready before the first GEMM
// loads any kernel file, so this gate is the only evidence of GPU readiness.
func gpuGate(engineDir string, probe gpuProbe) ([]string, error) {
	engineDir = filepath.Clean(engineDir)
	libraries := make([]enginelaunch.GPULibrary, len(gpuLibraries))
	for i, lib := range gpuLibraries {
		libraries[i] = enginelaunch.GPULibrary{Dir: libraryDir(engineDir, lib.name), EnvVar: lib.envVar, Extra: lib.extra}
	}
	env, err := enginelaunch.GPUPathGate(engineDir, libraries, probe)
	if errors.Is(err, enginelaunch.ErrKernelPathTooLong) {
		return nil, ErrGPUPathTooDeep
	}
	return env, err
}
