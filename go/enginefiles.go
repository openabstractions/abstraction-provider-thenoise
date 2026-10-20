package thenoise

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
)

// manifestHeader starts every engine manifest. The manifest lists each file
// of the pinned ZIP as "<path>\t<sha256>\t<size>", sorted by path, with LF
// line ends; directories are implied by file paths.
const manifestHeader = "# thenoise engine manifest v1 zip-sha256=" + EngineZIPSHA256 + "\n"

// manifestEntry is one regular file recorded from the verified ZIP.
type manifestEntry struct {
	Path   string // slash-separated, relative to the engine folder
	SHA256 string // lower-case hex
	Size   int64
}

// engineManifest is the RECORD-style list of every file of the pinned ZIP.
type engineManifest struct {
	entries []manifestEntry
}

func (m engineManifest) encode() []byte {
	var b bytes.Buffer
	b.WriteString(manifestHeader)
	for _, e := range m.entries {
		b.WriteString(e.Path)
		b.WriteByte('\t')
		b.WriteString(e.SHA256)
		b.WriteByte('\t')
		b.WriteString(strconv.FormatInt(e.Size, 10))
		b.WriteByte('\n')
	}
	return b.Bytes()
}

// files is the manifest in the shared provider kit's form.
func (m engineManifest) files() programfiles.Manifest {
	out := programfiles.Manifest{Files: make([]programfiles.File, len(m.entries))}
	for i, e := range m.entries {
		out.Files[i] = programfiles.File{Path: e.Path, Size: e.Size, SHA256: e.SHA256}
	}
	return out
}

func digestHex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// readManifest loads a manifest and accepts it only when its SHA-256 equals
// pin. The pin makes a rewritten manifest a refusal, whoever can write the
// provider's state folder.
func readManifest(file, pin string) (engineManifest, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return engineManifest{}, errors.New("thenoise: the engine has no verified manifest; install it from the pinned ZIP")
		}
		return engineManifest{}, fmt.Errorf("thenoise: reading engine manifest: %w", err)
	}
	if got := digestHex(data); got != pin {
		return engineManifest{}, fmt.Errorf("thenoise: engine manifest %s has SHA-256 %s, expected %s", file, got, pin)
	}
	return parseManifest(data)
}

func parseManifest(data []byte) (engineManifest, error) {
	if !bytes.HasPrefix(data, []byte(manifestHeader)) {
		return engineManifest{}, errors.New("thenoise: engine manifest header is not the pinned engine's")
	}
	var m engineManifest
	scanner := bufio.NewScanner(bytes.NewReader(data[len(manifestHeader):]))
	scanner.Buffer(make([]byte, 64<<10), 64<<10)
	for scanner.Scan() {
		fields := strings.Split(scanner.Text(), "\t")
		if len(fields) != 3 {
			return engineManifest{}, errors.New("thenoise: engine manifest line is malformed")
		}
		size, err := strconv.ParseInt(fields[2], 10, 64)
		if err != nil || size < 0 || len(fields[1]) != 64 {
			return engineManifest{}, errors.New("thenoise: engine manifest line is malformed")
		}
		m.entries = append(m.entries, manifestEntry{Path: fields[0], SHA256: fields[1], Size: size})
	}
	if err := scanner.Err(); err != nil {
		return engineManifest{}, err
	}
	return m, nil
}

// recordFile is the shared kit's metadata record, beside the manifest in the
// provider's private state.
func recordFile(manifestPath string) string {
	return filepath.Join(filepath.Dir(manifestPath), EngineFolderName+".files")
}

func engineTree(engineDir, manifestPath string, m engineManifest) programfiles.Tree {
	return programfiles.Tree{Name: "engine folder " + engineDir, Root: engineDir, Manifest: m.files(), Record: recordFile(manifestPath)}
}

// fromFiles is the shared kit's manifest in this provider's form.
func fromFiles(m programfiles.Manifest) engineManifest {
	out := engineManifest{entries: make([]manifestEntry, len(m.Files))}
	for i, f := range m.Files {
		out.entries[i] = manifestEntry{Path: f.Path, SHA256: f.SHA256, Size: f.Size}
	}
	sort.Slice(out.entries, func(i, j int) bool { return out.entries[i].Path < out.entries[j].Path })
	return out
}

// zipPrefix starts every entry name of the pinned ZIP, which also holds a
// "./" root entry.
const zipPrefix = "./"

// InstallEngine verifies the pinned ZIP against EngineZIPBytes and
// EngineZIPSHA256 before any entry is read and records its manifest, which
// must hash to EngineManifestSHA256. When engineDir is absent or empty it
// also extracts the ZIP there; an existing tree is kept and must equal the
// ZIP. The installed tree gets the shared kit's read-only folder ACL.
func InstallEngine(ctx context.Context, zipPath, engineDir, manifestPath string) error {
	return installEngine(ctx, programfiles.NewVerifier(programfiles.Options{}), zipPath, EngineZIPSHA256, EngineZIPBytes, engineDir, manifestPath, EngineManifestSHA256)
}

// UnlockEngine proves the pinned engine files and gives its read-only folders
// back to this account for explicit local maintenance. It starts no process and
// removes no files. Setup must prove its own journal ownership before calling it
// for removal; a matching engine alone does not establish that ownership.
func UnlockEngine(ctx context.Context, engineDir, manifestPath string) error {
	return unlockEngine(ctx, engineDir, manifestPath, EngineManifestSHA256)
}

func unlockEngine(ctx context.Context, engineDir, manifestPath, manifestPin string) error {
	if !filepath.IsAbs(engineDir) || !filepath.IsAbs(manifestPath) {
		return errors.New("thenoise: engine folder and manifest must be absolute paths")
	}
	m, err := readManifest(manifestPath, manifestPin)
	if err != nil {
		return err
	}
	// Verify checks the complete pinned tree without requiring every folder's
	// read-only ACL. UnlockWithin can finish an interrupted earlier unlock.
	if err := programfiles.Verify(ctx, "engine folder "+engineDir, engineDir, m.files()); err != nil {
		return err
	}
	return programfiles.UnlockWithin(engineDir)
}

// installEngine is the provider kit's archive install (inference
// CONTRACT.md INF-F6) of the pinned ZIP with the manifest recorded from it:
// every file entry is kept, and the recorded manifest must hash to
// manifestPin before the extracted tree replaces anything. The manifest file
// is written once the tree is installed.
func installEngine(ctx context.Context, v *programfiles.Verifier, zipPath, zipPin string, zipSize int64, engineDir, manifestPath, manifestPin string) error {
	if !filepath.IsAbs(zipPath) || !filepath.IsAbs(engineDir) || !filepath.IsAbs(manifestPath) {
		return errors.New("thenoise: engine ZIP, engine folder and manifest must be absolute paths")
	}
	extract, err := emptyOrAbsent(engineDir)
	if err != nil {
		return err
	}
	if extract && utf16Len(filepath.Clean(engineDir)) > MaxEngineDirLength {
		return fmt.Errorf("thenoise: engine folder %s is longer than %d characters; choose a shorter one with -engine-dir", engineDir, MaxEngineDirLength)
	}
	m, err := v.InstallFromArchive(ctx, programfiles.ArchiveInstall{
		Archive: programfiles.PinnedArchive{Path: zipPath, Size: zipSize, SHA256: zipPin},
		Tree:    programfiles.Tree{Name: "engine folder " + engineDir, Root: engineDir, Record: recordFile(manifestPath)},
		Prefix:  zipPrefix,
		Derive: func(m programfiles.Manifest) error {
			if got := digestHex(fromFiles(m).encode()); got != manifestPin {
				return fmt.Errorf("thenoise: manifest recorded from the ZIP has SHA-256 %s, expected %s", got, manifestPin)
			}
			return nil
		},
	})
	if err != nil {
		return fmt.Errorf("thenoise: engine ZIP %s: %w", zipPath, err)
	}
	return writeFileAtomic(manifestPath, fromFiles(m).encode())
}

func emptyOrAbsent(dir string) (bool, error) {
	entries, err := os.ReadDir(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return true, nil
	}
	if err != nil {
		return false, err
	}
	return len(entries) == 0, nil
}

func writeFileAtomic(file string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(file), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(file), filepath.Base(file)+".*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(name)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return err
	}
	if err := os.Rename(name, file); err != nil {
		os.Remove(name)
		return err
	}
	return nil
}

// launchEngine proves the installed engine against its pinned manifest
// through the shared kit (INF-F1..INF-F4) and returns the hold the launch
// keeps until the engine exits. A folder whose files all prove but whose
// read-only ACL is gone gets the ACL again, with one log line, and launches
// (programfiles LaunchRepairing). A changed, missing or extra file or folder,
// a link, or a folder whose ACL cannot be set again refuses the launch and
// names the reinstall step. A file another program holds so that the kit cannot
// open or read it refuses this launch only, with the system error and no
// reinstall step.
func launchEngine(ctx context.Context, v *programfiles.Verifier, engineDir, manifestPath, manifestPin string) (*programfiles.Hold, error) {
	m, err := readManifest(manifestPath, manifestPin)
	if err != nil {
		return nil, err
	}
	hold, _, err := v.LaunchRepairing(ctx, engineTree(engineDir, manifestPath, m), log.Writer())
	if errors.Is(err, programfiles.ErrFolderACL) {
		return nil, fmt.Errorf("thenoise: %w: delete the engine manifest under -state and start the provider with -engine-zip", err)
	}
	if errors.Is(err, programfiles.ErrUnreadable) {
		return nil, fmt.Errorf("thenoise: %w: another program holds an engine file; try again once it lets go", err)
	}
	if errors.Is(err, programfiles.ErrMismatch) {
		return nil, fmt.Errorf("thenoise: %w: delete the engine manifest under -state and start the provider with -engine-zip to install the engine again", err)
	}
	if err != nil {
		return nil, fmt.Errorf("thenoise: %w", err)
	}
	return hold, nil
}
