package thenoise

import (
	"archive/zip"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
)

func TestInstallExtractsTheVerifiedZIPAndRecordsItsManifest(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	data, err := os.ReadFile(filepath.Join(engineDir, "Lib", "site-packages", "a.py"))
	if err != nil || string(data) != "print(1)\n" {
		t.Fatalf("extracted file: %q %v", data, err)
	}
	if err := verifyEngine(t.Context(), engineDir, manifestFile(stateDir), pin); err != nil {
		t.Fatalf("fresh install failed verification: %v", err)
	}
	if _, err := os.Stat(engineDir + ".partial"); !os.IsNotExist(err) {
		t.Fatalf("staging folder left behind: %v", err)
	}
}

func TestUnlockProvesEngineAndPreservesItsFiles(t *testing.T) {
	engine, state, pin := installSmall(t)
	file := filepath.Join(engine, "Lib", "site-packages", "a.py")
	before, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := unlockEngine(t.Context(), engine, manifestFile(state), pin); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(file)
	if err != nil || string(after) != string(before) {
		t.Fatalf("unlock changed engine files: %q %v", after, err)
	}
}

func TestUnlockRefusesUnverifiedEngineAndManifest(t *testing.T) {
	engine, state, pin := installSmall(t)
	manifest := manifestFile(state)
	for _, args := range [][3]string{{"relative", manifest, pin}, {engine, "relative", pin},
		{engine, manifest, strings.Repeat("0", 64)}, {t.TempDir(), manifest, pin}} {
		if err := unlockEngine(t.Context(), args[0], args[1], args[2]); err == nil {
			t.Fatalf("unverified input accepted: %q", args)
		}
	}
	file := filepath.Join(engine, "Lib", "site-packages", "a.py")
	if err := os.WriteFile(file, []byte("changed"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := unlockEngine(t.Context(), engine, manifest, pin); err == nil {
		t.Fatal("changed engine was unlocked")
	}
}

func TestInstallRefusesAZIPWithAnotherHashBeforeExtracting(t *testing.T) {
	base := shortTempDir(t)
	zipPath := filepath.Join(base, "engine.zip")
	writeZIP(t, zipPath, smallBundle())
	pin := manifestPinOf(t, zipPath)
	engineDir := filepath.Join(base, "e")
	err := installEngine(t.Context(), testFiles(), zipPath, strings.Repeat("0", 64), sizeOf(t, zipPath), engineDir, manifestFile(filepath.Join(base, "state")), pin)
	if err == nil || !strings.Contains(err.Error(), "SHA-256") {
		t.Fatalf("wrong ZIP accepted: %v", err)
	}
	if _, err := os.Stat(engineDir); !os.IsNotExist(err) {
		t.Fatal("a refused ZIP was extracted")
	}
}

func TestInstallRefusesAManifestOtherThanThePinnedOne(t *testing.T) {
	base := shortTempDir(t)
	zipPath := filepath.Join(base, "engine.zip")
	zipPin := writeZIP(t, zipPath, smallBundle())
	engineDir := filepath.Join(base, "e")
	err := installEngine(t.Context(), testFiles(), zipPath, zipPin, sizeOf(t, zipPath), engineDir, manifestFile(filepath.Join(base, "state")), strings.Repeat("0", 64))
	if err == nil {
		t.Fatal("a ZIP whose manifest differs from the pin was installed")
	}
	if _, err := os.Stat(engineDir); !os.IsNotExist(err) {
		t.Fatal("a refused ZIP left an engine folder")
	}
}

func TestInstallRefusesEscapingEntries(t *testing.T) {
	for _, bad := range []string{"../evil.dll", "./../evil.dll", "/abs.dll", "a/../../evil.dll", `a\..\evil.dll`, "C:/evil.dll",
		"a:stream", "NUL", "com1.dll", "Lib/aux", "Lib/Lpt9.tar.gz", "./con"} {
		for _, name := range []string{bad, "./" + bad} {
			base := shortTempDir(t)
			zipPath := filepath.Join(base, "engine.zip")
			f, err := os.Create(zipPath)
			if err != nil {
				t.Fatal(err)
			}
			w := zip.NewWriter(f)
			entry, err := w.Create(name)
			if err != nil {
				t.Fatal(err)
			}
			entry.Write([]byte("x"))
			w.Close()
			f.Close()
			data, err := os.ReadFile(zipPath)
			if err != nil {
				t.Fatal(err)
			}
			engineDir := filepath.Join(base, "e")
			err = installEngine(t.Context(), testFiles(), zipPath, digestHex(data), int64(len(data)), engineDir, manifestFile(filepath.Join(base, "state")), strings.Repeat("0", 64))
			if !errors.Is(err, programfiles.ErrArchive) {
				t.Fatalf("entry %q: %v, wanted the kit's archive refusal", name, err)
			}
			if _, statErr := os.Stat(filepath.Join(base, "evil.dll")); statErr == nil {
				t.Fatalf("entry %q escaped the engine folder", name)
			}
			if _, statErr := os.Stat(engineDir); !os.IsNotExist(statErr) {
				t.Fatalf("entry %q left an engine folder", name)
			}
		}
	}
}

func TestInstallRefusesAnEngineFolderLongerThanTheLimit(t *testing.T) {
	base := shortTempDir(t)
	zipPath := filepath.Join(base, "engine.zip")
	zipPin := writeZIP(t, zipPath, smallBundle())
	pin := manifestPinOf(t, zipPath)
	engineDir := filepath.Join(base, "e")
	if pad := MaxEngineDirLength + 1 - utf16Len(engineDir); pad > 0 {
		engineDir += strings.Repeat("x", pad)
	}
	err := installEngine(t.Context(), testFiles(), zipPath, zipPin, sizeOf(t, zipPath), engineDir, manifestFile(filepath.Join(base, "state")), pin)
	if err == nil || !strings.Contains(err.Error(), "longer than 87") || !strings.Contains(err.Error(), "-engine-dir") {
		t.Fatalf("an %d-character engine folder was accepted or the refusal names no flag: %v", utf16Len(engineDir), err)
	}
}

func TestDefaultEngineDirIsTheShortOARoot(t *testing.T) {
	if filepath.Separator != '\\' {
		t.Skip("Windows paths")
	}
	local := filepath.Join(`C:\Users`, strings.Repeat("n", 38), "AppData", "Local")
	t.Setenv("LOCALAPPDATA", local)
	dir, err := DefaultEngineDir()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(local, "OA", "engines", "thenoise-0.9.1"); dir != want {
		t.Fatalf("default engine folder %s, want %s", dir, want)
	}
	if utf16Len(dir) != MaxEngineDirLength {
		t.Fatalf("a 38-character profile name gives a %d-character default, want %d", utf16Len(dir), MaxEngineDirLength)
	}
}

func TestVerificationRefusesAChangedFile(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	file := filepath.Join(engineDir, "Lib", "site-packages", "a.py")
	if err := os.WriteFile(file, []byte("print(2)\n"), 0o644); err != nil { // same size
		t.Fatal(err)
	}
	err := verifyEngine(t.Context(), engineDir, manifestFile(stateDir), pin)
	if err == nil || !strings.Contains(err.Error(), "Lib/site-packages/a.py") {
		t.Fatalf("changed file passed: %v", err)
	}
}

func TestVerificationRefusesAnExtraFile(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	extra := filepath.Join(engineDir, "Lib", "site-packages", "__pycache__", "a.cpython-314.pyc")
	if err := os.MkdirAll(filepath.Dir(extra), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(extra, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := verifyEngine(t.Context(), engineDir, manifestFile(stateDir), pin)
	if err == nil || !strings.Contains(err.Error(), "Lib/site-packages/__pycache__") {
		t.Fatalf("extra folder with a compiled file passed: %v", err)
	}
	if err := os.Rename(extra, filepath.Join(engineDir, "Lib", "site-packages", "a.cpython-314.pyc")); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Dir(extra)); err != nil {
		t.Fatal(err)
	}
	err = verifyEngine(t.Context(), engineDir, manifestFile(stateDir), pin)
	if err == nil || !strings.Contains(err.Error(), "a.cpython-314.pyc") {
		t.Fatalf("extra file passed: %v", err)
	}
}

// INF-F1..INF-F4 through the provider: the first launch of a process hashes
// in full, later ones check metadata, a same-size change in place is caught,
// and a folder whose read-only ACL cannot be set again refuses the launch.
func TestLaunchUsesTheSharedEngineFileKit(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	v := testFiles()
	// The install's record proves the tree for this and every later
	// provider process (Decision 2, amendment 2).
	for i := 0; i < 2; i++ {
		h, err := launchEngine(t.Context(), v, engineDir, manifestFile(stateDir), pin)
		if err != nil || h.Full() {
			t.Fatalf("launch %d after the install: full=%v %v", i+1, h.Full(), err)
		}
		h.Release()
	}
	if _, err := os.Stat(recordFile(manifestFile(stateDir))); err != nil {
		t.Fatalf("no metadata record beside the manifest: %v", err)
	}
	original, err := os.ReadFile(filepath.Join(engineDir, "thenoise.bat"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(engineDir, "thenoise.bat"), []byte("@echo  on\r\n"), 0o644); err != nil { // same size
		t.Fatal(err)
	}
	if _, err := launchEngine(t.Context(), v, engineDir, manifestFile(stateDir), pin); !errors.Is(err, programfiles.ErrMismatch) || !strings.Contains(err.Error(), "thenoise.bat") {
		t.Fatalf("a same-size launcher change passed the metadata launch: %v", err)
	}
	if err := os.WriteFile(filepath.Join(engineDir, "thenoise.bat"), original, 0o644); err != nil {
		t.Fatal(err)
	}
	// A folder whose ACL cannot be set again refuses with the remedy.
	refusing := programfiles.NewVerifier(programfiles.Options{ACL: testACL{refuse: true}})
	_, err = launchEngine(t.Context(), refusing, engineDir, manifestFile(stateDir), pin)
	if !errors.Is(err, programfiles.ErrFolderACL) || !strings.Contains(err.Error(), "delete the engine manifest under -state") || !strings.Contains(err.Error(), "-engine-zip") {
		t.Fatalf("a folder without the read-only ACL launched, or the refusal names no remedy: %v", err)
	}
}

func TestVerificationRefusesAMissingFile(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	if err := os.Remove(filepath.Join(engineDir, "thenoise.bat")); err != nil {
		t.Fatal(err)
	}
	if err := verifyEngine(t.Context(), engineDir, manifestFile(stateDir), pin); err == nil || !strings.Contains(err.Error(), "thenoise.bat is missing") {
		t.Fatalf("missing file passed: %v", err)
	}
}

// A deleted engine folder is a missing tree: the launch refusal is the
// kit's mismatch and names the reinstall step.
func TestLaunchOfADeletedEngineFolderNamesTheReinstall(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	if err := os.RemoveAll(engineDir); err != nil {
		t.Fatal(err)
	}
	_, err := launchEngine(t.Context(), testFiles(), engineDir, manifestFile(stateDir), pin)
	var mismatch *programfiles.MismatchError
	if !errors.As(err, &mismatch) || mismatch.Kind != programfiles.Missing || mismatch.Path != "." || errors.Is(err, programfiles.ErrUnreadable) ||
		!strings.Contains(err.Error(), "delete the engine manifest under -state") || !strings.Contains(err.Error(), "-engine-zip") {
		t.Fatalf("a deleted engine folder: %v", err)
	}
}

func TestVerificationRefusesARewrittenManifest(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	file := filepath.Join(engineDir, "Lib", "site-packages", "a.py")
	if err := os.WriteFile(file, []byte("print('changed')\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Record the changed tree in the manifest, as a writer of the state
	// folder could.
	data, err := os.ReadFile(manifestFile(stateDir))
	if err != nil {
		t.Fatal(err)
	}
	m, err := parseManifest(data)
	if err != nil {
		t.Fatal(err)
	}
	for i := range m.entries {
		if m.entries[i].Path == "Lib/site-packages/a.py" {
			content, _ := os.ReadFile(file)
			m.entries[i].SHA256 = digestHex(content)
			m.entries[i].Size = int64(len(content))
		}
	}
	if err := os.WriteFile(manifestFile(stateDir), m.encode(), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := verifyEngine(t.Context(), engineDir, manifestFile(stateDir), pin); err == nil {
		t.Fatal("a rewritten manifest vouched for a changed file")
	}
}

func TestVerificationRefusesWithoutAManifest(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	if err := os.Remove(manifestFile(stateDir)); err != nil {
		t.Fatal(err)
	}
	if err := verifyEngine(t.Context(), engineDir, manifestFile(stateDir), pin); err == nil {
		t.Fatal("an engine without a manifest passed")
	}
}

func TestInstallKeepsAnExistingMatchingTree(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	base := filepath.Dir(engineDir)
	zipPath := filepath.Join(base, "engine.zip")
	zipPin := writeZIP(t, zipPath, smallBundle())
	if err := os.Remove(manifestFile(stateDir)); err != nil {
		t.Fatal(err)
	}
	if err := installEngine(t.Context(), testFiles(), zipPath, zipPin, sizeOf(t, zipPath), engineDir, manifestFile(stateDir), pin); err != nil {
		t.Fatalf("an existing tree equal to the ZIP was refused: %v", err)
	}
	if err := os.WriteFile(filepath.Join(engineDir, "stray.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(manifestFile(stateDir)); err != nil {
		t.Fatal(err)
	}
	if err := installEngine(t.Context(), testFiles(), zipPath, zipPin, sizeOf(t, zipPath), engineDir, manifestFile(stateDir), pin); err == nil {
		t.Fatal("an existing tree with a stray file was accepted")
	}
}
