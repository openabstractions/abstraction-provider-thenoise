//go:build windows

package thenoise

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestGPUGateRealShortNames runs the gate with the real GetShortPathNameW and
// GetFinalPathNameByHandleW on a deep folder. It skips on a volume that
// generates no 8.3 names.
func TestGPUGateRealShortNames(t *testing.T) {
	engineDir := filepath.Join(t.TempDir(), "averyveryverylongfolderthatgetsashortname", "anotherlongfoldername_for_83_names")
	hip := gpuLibraries[0]
	writeKernels(t, engineDir, []string{nameForTotal(t, libraryDir(engineDir, hip.name), 1, maxLibraryPath+30)}, nil)
	probe := systemGPUProbe()
	probe.LongPathsEnabled = func() bool { return false }
	shorts := map[string]string{}
	for _, lib := range gpuLibraries {
		long := libraryDir(engineDir, lib.name)
		short, err := systemGPUProbe().ShortPath(long)
		if err != nil {
			t.Fatal(err)
		}
		if utf16Len(short) > utf16Len(long)-40 {
			t.Skipf("volume gives no useful 8.3 names: %s", short)
		}
		shorts[lib.envVar] = short
	}
	env, err := gpuGate(engineDir, probe)
	if err != nil {
		t.Fatal(err)
	}
	for _, kv := range env {
		name, value, _ := strings.Cut(kv, "=")
		if shorts[name] != value {
			t.Fatalf("%s=%s, want %s", name, value, shorts[name])
		}
	}
	if len(env) != 2 {
		t.Fatalf("fallback environment %q", env)
	}
}

// TestRealJunctionIsSeenAndResolvesOutside shows the two Windows answers the
// gate's condition 9 checks rely on: a junction carries the reparse
// attribute, and its final path names the target outside the bundle.
func TestRealJunctionIsSeenAndResolvesOutside(t *testing.T) {
	base := t.TempDir()
	engineDir := filepath.Join(base, "e")
	writeKernels(t, engineDir, nil, nil)
	outside := filepath.Join(base, "outside", "library")
	if err := os.MkdirAll(filepath.Join(outside, "gfx1151"), 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Dir(libraryDir(engineDir, "hipblaslt"))
	if err := os.RemoveAll(link); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(filepath.Join(os.Getenv("SystemRoot"), "System32", "cmd.exe"), "/d", "/c", "mklink", "/J", link, outside).CombinedOutput(); err != nil {
		t.Skipf("cannot create a junction: %v %s", err, out)
	}
	defer os.Remove(link)
	reparse, err := systemGPUProbe().ReparsePoint(link)
	if err != nil || !reparse {
		t.Fatalf("junction not reported as a reparse point: %v %v", reparse, err)
	}
	if plain, err := systemGPUProbe().ReparsePoint(engineDir); err != nil || plain {
		t.Fatalf("plain folder reported as a reparse point: %v %v", plain, err)
	}
	bundle, err := systemGPUProbe().FinalPath(engineDir)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := systemGPUProbe().FinalPath(filepath.Join(link, "gfx1151"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(strings.ToLower(resolved), strings.ToLower(bundle)) {
		t.Fatalf("junction target %s resolved inside the bundle %s", resolved, bundle)
	}
	// The gate refuses the junction on the way when it needs 8.3 names: a
	// probe with long paths off and kernel paths over the limit.
	writeKernels(t, engineDir, nil, []string{nameForTotal(t, libraryDir(engineDir, "rocblas"), 2, maxLibraryPath+10)})
	probe := systemGPUProbe()
	probe.LongPathsEnabled = func() bool { return false }
	if _, err := gpuGate(engineDir, probe); !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("junction on the way passed: %v", err)
	}
}
