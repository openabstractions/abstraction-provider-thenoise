package thenoise

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-facade/go/supervision"
	"github.com/openabstractions/abstraction-provider-kit/go/enginelaunch"
)

// nameForTotal returns a kernel file name whose opened path below gfxDir is
// exactly total characters long when the library inserts extra separators.
// Tests pass the measured offsets as literals: hipBLASLt 1, rocBLAS 2.
func nameForTotal(t *testing.T, gfxDir string, extra, total int) string {
	t.Helper()
	n := total - utf16Len(gfxDir) - 1 - extra
	const prefix = "TensileLibrary_"
	if n < len(prefix)+5 || n > 250 {
		t.Skipf("temporary folder length %d leaves no room for a %d-character path", utf16Len(gfxDir), total)
	}
	return prefix + strings.Repeat("x", n-len(prefix)-4) + ".dat"
}

// writeKernels creates the two gfx1151 folders below engineDir with the named
// files; an empty list gets one short file.
func writeKernels(t *testing.T, engineDir string, hip, roc []string) {
	t.Helper()
	for i, names := range [][]string{hip, roc} {
		dir := libraryDir(engineDir, gpuLibraries[i].name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if len(names) == 0 {
			names = []string{"TensileLibrary_a.dat"}
		}
		for _, name := range names {
			if err := os.WriteFile(filepath.Join(dir, name), []byte("k"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
}

// fakeProbe answers the gate with long paths off, no 8.3 names, final paths
// equal to the input and no reparse points.
func fakeProbe() gpuProbe {
	return gpuProbe{
		LongPathsEnabled: func() bool { return false },
		ShortPath:        func(string) (string, error) { return "", errors.New("no 8.3 names") },
		FinalPath:        func(p string) (string, error) { return `\\?\` + p, nil },
		ReparsePoint:     func(string) (bool, error) { return false, nil },
	}
}

// kernelFiles and longestOpen read a bundle's kernel folders through the
// provider kit's gate, for the measurements the tests log.
func kernelFiles(dir string) ([]string, error) { return enginelaunch.KernelFiles(dir) }

func longestOpen(base string, lib gpuLibrary, files []string) int {
	return supervision.LongestOpen(base, lib.extra, files)
}

// writeZIP writes a ZIP shaped like the pinned one: every entry named
// "./<path>", with a "./" root and folder entries. It returns its SHA-256.
func writeZIP(t *testing.T, file string, files map[string]string) string {
	t.Helper()
	f, err := os.Create(file)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(f)
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	if _, err := w.Create("./"); err != nil {
		t.Fatal(err)
	}
	dirs := map[string]bool{}
	for _, name := range names {
		for d := filepath.ToSlash(filepath.Dir(name)); d != "." && !dirs[d]; d = filepath.ToSlash(filepath.Dir(d)) {
			dirs[d] = true
			if _, err := w.Create("./" + d + "/"); err != nil {
				t.Fatal(err)
			}
		}
		entry, err := w.Create("./" + name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(files[name])); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// sizeOf is a test ZIP's size, its size pin.
func sizeOf(t *testing.T, file string) int64 {
	t.Helper()
	info, err := os.Stat(file)
	if err != nil {
		t.Fatal(err)
	}
	return info.Size()
}

// manifestPinOf records the manifest of a ZIP the way the provider does and
// returns its SHA-256.
func manifestPinOf(t *testing.T, file string) string {
	t.Helper()
	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	m, err := programfiles.ArchiveManifest(t.Context(), programfiles.PinnedArchive{Path: file, Size: int64(len(data)), SHA256: digestHex(data)}, zipPrefix)
	if err != nil {
		t.Fatal(err)
	}
	return digestHex(fromFiles(m).encode())
}

// shortTempDir is a temporary folder whose path leaves room below the
// 87-character engine folder limit; t.TempDir adds the test's name.
func shortTempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "tn")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

// smallBundle is a miniature engine: the launcher, one Python file and one
// kernel file in each ROCm library folder.
func smallBundle() map[string]string {
	return map[string]string{
		"thenoise.bat":           "@echo off\r\n",
		"Lib/site-packages/a.py": "print(1)\n",
		"Lib/site-packages/_rocm_sdk_libraries/bin/hipblaslt/library/gfx1151/TensileLibrary_h.dat": "h",
		"Lib/site-packages/_rocm_sdk_libraries/bin/rocblas/library/gfx1151/TensileLibrary_r.dat":   "r",
	}
}

// testACL stands for the shared kit's read-only folder ACL, whose own tests
// set and check the real one; refuse makes every folder fail the check.
type testACL struct{ refuse bool }

func (testACL) Restrict([]string) error { return nil }

func (a testACL) Check(folder string, _ *programfiles.Handle) error {
	if a.refuse {
		return fmt.Errorf("%w: %s", programfiles.ErrFolderACL, folder)
	}
	return nil
}

// testFiles is a fresh provider process's verifier: its first launch of a
// tree hashes every file.
func testFiles() *programfiles.Verifier {
	return programfiles.NewVerifier(programfiles.Options{ACL: testACL{}})
}

// verifyEngine proves the installed engine as a fresh provider process's
// first launch does and releases the hold.
func verifyEngine(ctx context.Context, engineDir, manifestPath, pin string) error {
	h, err := launchEngine(ctx, testFiles(), engineDir, manifestPath, pin)
	if err != nil {
		return err
	}
	return h.Release()
}

// installSmall writes the miniature ZIP, installs it into a fresh engine
// folder and returns the engine folder, state folder and manifest pin.
func installSmall(t *testing.T) (engineDir, stateDir, pin string) {
	t.Helper()
	base := shortTempDir(t)
	zipPath := filepath.Join(base, "engine.zip")
	zipPin := writeZIP(t, zipPath, smallBundle())
	pin = manifestPinOf(t, zipPath)
	engineDir = filepath.Join(base, "e")
	stateDir = filepath.Join(base, "state")
	if err := installEngine(t.Context(), testFiles(), zipPath, zipPin, sizeOf(t, zipPath), engineDir, manifestFile(stateDir), pin); err != nil {
		t.Fatal(err)
	}
	return engineDir, stateDir, pin
}
