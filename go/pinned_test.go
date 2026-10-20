package thenoise

import (
	"context"
	"os"
	"testing"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
)

// TestPinnedManifestFromLocalZIP recomputes EngineManifestSHA256 from a local
// copy of the pinned ZIP named by OA_THENOISE_ZIP, and verifies an extracted
// tree named by OA_THENOISE_TREE against it. It reads about 1.6 GB (and the
// tree's 5 GB); without the variables it is skipped.
func TestPinnedManifestFromLocalZIP(t *testing.T) {
	zipPath := os.Getenv("OA_THENOISE_ZIP")
	if zipPath == "" {
		t.Skip("OA_THENOISE_ZIP names no local copy of the pinned ZIP; the owner's copy is .build/headless-providers/thenoise/thenoise-0.9.1-rocm10.1.0-gfx1151-win-x64.zip in the main checkout")
	}
	// The kit refuses a ZIP of another size or SHA-256 before it reads an
	// entry, as the install does.
	files, err := enginefiles.ArchiveManifest(context.Background(), enginefiles.PinnedArchive{Path: zipPath, Size: EngineZIPBytes, SHA256: EngineZIPSHA256}, zipPrefix)
	if err != nil {
		t.Fatal(err)
	}
	m := fromFiles(files)
	encoded := m.encode()
	t.Logf("manifest: %d files, %d bytes, SHA-256 %s", len(m.entries), len(encoded), digestHex(encoded))
	if got := digestHex(encoded); got != EngineManifestSHA256 {
		t.Fatalf("manifest SHA-256 %s, pinned %s", got, EngineManifestSHA256)
	}
	if tree := os.Getenv("OA_THENOISE_TREE"); tree != "" {
		if err := enginefiles.Verify(context.Background(), "thenoise", tree, m.files()); err != nil {
			t.Fatalf("extracted tree %s: %v", tree, err)
		}
	}
}

// TestGateOnLocalTree applies the gate, with long paths treated as off, to an
// extraction of the pinned ZIP named by OA_THENOISE_TREE. The measured limit
// says it passes exactly when the folder is at most MaxEngineDirLength long.
func TestGateOnLocalTree(t *testing.T) {
	tree := os.Getenv("OA_THENOISE_TREE")
	if tree == "" {
		t.Skip("OA_THENOISE_TREE names no extraction of the pinned ZIP")
	}
	for _, lib := range gpuLibraries {
		files, err := kernelFiles(libraryDir(tree, lib.name))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %d files, longest opened path %d at folder length %d", lib.name, len(files), longestOpen(libraryDir(tree, lib.name), lib, files), utf16Len(tree))
	}
	probe := fakeProbe()
	_, err := gpuGate(tree, probe)
	if fits := utf16Len(tree) <= MaxEngineDirLength; fits != (err == nil) {
		t.Fatalf("folder length %d, gate result %v", utf16Len(tree), err)
	}
}
