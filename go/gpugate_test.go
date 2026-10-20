package thenoise

import (
	"errors"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestGPUGateHipBLASLtLimitIsExact(t *testing.T) {
	engineDir := filepath.Join(t.TempDir(), "e")
	hip := gpuLibraries[0]
	gfx := libraryDir(engineDir, hip.name)
	writeKernels(t, engineDir, []string{nameForTotal(t, gfx, 1, maxLibraryPath)}, nil)
	env, err := gpuGate(engineDir, fakeProbe())
	if err != nil || env != nil {
		t.Fatalf("a 259-character hipBLASLt path failed the gate: env=%v err=%v", env, err)
	}
	writeKernels(t, engineDir, []string{nameForTotal(t, gfx, 1, maxLibraryPath+1)}, nil)
	if _, err := gpuGate(engineDir, fakeProbe()); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("a 260-character hipBLASLt path passed the gate: %v", err)
	}
}

func TestGPUGateRocBLASAddsTwoSeparators(t *testing.T) {
	engineDir := filepath.Join(t.TempDir(), "e")
	roc := gpuLibraries[1]
	gfx := libraryDir(engineDir, roc.name)
	writeKernels(t, engineDir, nil, []string{nameForTotal(t, gfx, 2, maxLibraryPath)})
	if _, err := gpuGate(engineDir, fakeProbe()); err != nil {
		t.Fatalf("a 259-character rocBLAS path failed the gate: %v", err)
	}
	writeKernels(t, engineDir, nil, []string{nameForTotal(t, gfx, 2, maxLibraryPath+1)})
	if _, err := gpuGate(engineDir, fakeProbe()); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("a 260-character rocBLAS path passed the gate: %v", err)
	}
}

func TestGPUGateCoversEveryFileInTheFolder(t *testing.T) {
	engineDir := filepath.Join(t.TempDir(), "e")
	hip := gpuLibraries[0]
	gfx := libraryDir(engineDir, hip.name)
	// Same length, the prefix of the bundle's Kernels.so-000-gfx1151.hsaco.
	long := "Kernels.so-000-" + strings.TrimPrefix(nameForTotal(t, gfx, 1, maxLibraryPath+1), "TensileLibrary_")
	writeKernels(t, engineDir, []string{"TensileLibrary_a.dat", long}, nil)
	if _, err := gpuGate(engineDir, fakeProbe()); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("a file without the TensileLibrary_ prefix escaped the gate: %v", err)
	}
}

func TestGPUGateMessageIsThePlainSentence(t *testing.T) {
	const want = "The TheNoise engine folder is too deep for its GPU libraries; move it closer to the drive root."
	if ErrGPUPathTooDeep.Error() != want {
		t.Fatalf("refusal reads %q", ErrGPUPathTooDeep.Error())
	}
}

func TestGPUGatePassesDeepFolderWithLongPathsEnabled(t *testing.T) {
	engineDir := filepath.Join(t.TempDir(), "e")
	hip := gpuLibraries[0]
	writeKernels(t, engineDir, []string{nameForTotal(t, libraryDir(engineDir, hip.name), 1, maxLibraryPath+20)}, nil)
	probe := fakeProbe()
	probe.LongPathsEnabled = func() bool { return true }
	env, err := gpuGate(engineDir, probe)
	if err != nil || env != nil {
		t.Fatalf("long paths on: env=%v err=%v", env, err)
	}
}

// deepBundle builds a bundle whose hipBLASLt path is 20 characters over the
// limit and a probe whose 8.3 names are 30 characters shorter and resolve to
// the same folders.
func deepBundle(t *testing.T) (string, gpuProbe, map[string]string) {
	t.Helper()
	engineDir := filepath.Join(t.TempDir(), "e")
	hip := gpuLibraries[0]
	writeKernels(t, engineDir, []string{nameForTotal(t, libraryDir(engineDir, hip.name), 1, maxLibraryPath+20)}, nil)
	short := map[string]string{}
	for _, lib := range gpuLibraries {
		long := libraryDir(engineDir, lib.name)
		short[long] = long[:len(long)-40] + strings.Repeat("s", 10)
	}
	probe := fakeProbe()
	probe.ShortPath = func(p string) (string, error) {
		if s, ok := short[p]; ok {
			return s, nil
		}
		return p, nil
	}
	probe.FinalPath = func(p string) (string, error) {
		for long, s := range short {
			if p == s {
				return `\\?\` + long, nil
			}
		}
		return `\\?\` + p, nil
	}
	return engineDir, probe, short
}

func TestGPUGateUsesVerifiedShortNamesForBothLibraries(t *testing.T) {
	if runtime.GOOS != "windows" {
		// The probe answers in Windows final-path form (\\?\).
		t.Skip("8.3 names exist only on Windows, where the pinned win-x64 engine runs")
	}
	engineDir, probe, short := deepBundle(t)
	env, err := gpuGate(engineDir, probe)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"HIPBLASLT_TENSILE_LIBPATH=" + short[libraryDir(engineDir, "hipblaslt")],
		"ROCBLAS_TENSILE_LIBPATH=" + short[libraryDir(engineDir, "rocblas")],
	}
	if strings.Join(env, "\n") != strings.Join(want, "\n") {
		t.Fatalf("fallback environment %q, want %q", env, want)
	}
}

func TestGPUGateRefusesShortNamesThatDoNotHelp(t *testing.T) {
	engineDir, probe, _ := deepBundle(t)
	probe.ShortPath = func(p string) (string, error) { return p, nil } // volume without 8.3 names
	if _, err := gpuGate(engineDir, probe); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("an unchanged long path was accepted as its 8.3 name: %v", err)
	}
	// rocBLAS's long path fits, but it has no 8.3 name: both variables must
	// be 8.3 forms, so the fallback is refused.
	engineDir, probe, _ = deepBundle(t)
	withShort := probe.ShortPath
	probe.ShortPath = func(p string) (string, error) {
		if p == libraryDir(engineDir, "rocblas") {
			return p, nil
		}
		return withShort(p)
	}
	if env, err := gpuGate(engineDir, probe); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("a long path was used as an 8.3 name: %v %v", env, err)
	}
	engineDir, probe, short := deepBundle(t)
	inner := probe.ShortPath
	probe.ShortPath = func(p string) (string, error) {
		s, err := inner(p)
		return s + strings.Repeat("y", 29), err // still shorter, still over the limit
	}
	probe.FinalPath = func(p string) (string, error) {
		for long, s := range short {
			if strings.HasPrefix(p, s) {
				return `\\?\` + long, nil
			}
		}
		return `\\?\` + p, nil
	}
	if _, err := gpuGate(engineDir, probe); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("an 8.3 name that still fails the gate was used: %v", err)
	}
}

func TestGPUGateRefusesShortNameResolvingOutsideTheBundle(t *testing.T) {
	engineDir, probe, short := deepBundle(t)
	roc := short[libraryDir(engineDir, "rocblas")]
	inner := probe.FinalPath
	probe.FinalPath = func(p string) (string, error) {
		if p == roc {
			return `\\?\C:\elsewhere\library\gfx1151`, nil
		}
		return inner(p)
	}
	if _, err := gpuGate(engineDir, probe); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("an 8.3 name resolving outside the bundle was used: %v", err)
	}
}

func TestGPUGateRefusesReparsePointOnTheWay(t *testing.T) {
	engineDir, probe, _ := deepBundle(t)
	link := filepath.Dir(libraryDir(engineDir, "rocblas")) // ...\rocblas\library
	probe.ReparsePoint = func(p string) (bool, error) { return p == link, nil }
	if _, err := gpuGate(engineDir, probe); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("a reparse point under the bundle was accepted: %v", err)
	}
	probe.ReparsePoint = func(p string) (bool, error) { return p == engineDir, nil }
	if _, err := gpuGate(engineDir, probe); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("a reparse point at the bundle folder was accepted: %v", err)
	}
}
