package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-identity/placetest"
	thenoise "github.com/openabstractions/abstraction-provider-thenoise/go"
)

func TestEngineMaintenanceDispatchesWithoutServing(t *testing.T) {
	root := t.TempDir()
	engine, state, archive := filepath.Join(root, "engine"), filepath.Join(root, "state"), filepath.Join(root, "engine.zip")
	manifest := filepath.Join(state, thenoise.EngineFolderName+".manifest")
	for _, action := range []string{"install", "unlock"} {
		t.Run(action, func(t *testing.T) {
			var got []string
			install := func(_ context.Context, zip, dir, record string) error {
				got = []string{"install", zip, dir, record}
				return nil
			}
			unlock := func(_ context.Context, dir, record string) error {
				got = []string{"unlock", dir, record}
				return nil
			}
			args := []string{action, "-engine-dir", engine, "-state", state}
			want := []string{action, engine, manifest}
			if action == "install" {
				args = append(args, "-engine-zip", archive)
				want = []string{action, archive, engine, manifest}
			}
			if err := runEngine(t.Context(), args, install, unlock); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("dispatched %q, want %q", got, want)
			}
			if _, err := os.Stat(engine); !os.IsNotExist(err) {
				t.Fatalf("dispatch created an engine: %v", err)
			}
		})
	}
}

func TestEngineMaintenanceRefusesInvalidArgumentsBeforeEffects(t *testing.T) {
	root := t.TempDir()
	base := []string{"-engine-dir", filepath.Join(root, "engine"), "-state", filepath.Join(root, "state")}
	cases := [][]string{
		{}, {"remove"}, {"install"}, {"unlock", "-engine-dir", "relative", "-state", root},
		{"unlock", "-engine-dir", root, "-state", "relative"},
		append(append([]string{"install"}, base...), "-engine-zip", "relative"),
		append(append([]string{"unlock"}, base...), "-engine-zip", filepath.Join(root, "engine.zip")),
		append(append([]string{"unlock"}, base...), "extra"),
		append(append([]string{"install"}, base...), "-runtime", root),
	}
	for _, args := range cases {
		called := false
		err := runEngine(t.Context(), args,
			func(context.Context, string, string, string) error { called = true; return nil },
			func(context.Context, string, string) error { called = true; return nil })
		if err == nil || called {
			t.Fatalf("args %q: err=%v effects=%v", args, err, called)
		}
	}
}

func TestEngineInstallDispatchRefusesUnpinnedArchiveWithoutLaunch(t *testing.T) {
	root, err := os.MkdirTemp("", "oa-thenoise-cli-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(root); err != nil {
			t.Error(err)
		}
	})
	// The fixture must reach the ZIP guard even below a nested development
	// TEMP. Keep the same existing directory and the real engine path limit.
	if short, shortErr := placetest.ShortName(root); shortErr == nil {
		original, err := os.Stat(root)
		if err != nil {
			t.Fatal(err)
		}
		alias, err := os.Stat(short)
		if err != nil || !os.SameFile(original, alias) {
			t.Fatalf("fixture TEMP identity: %v", err)
		}
		root = short
	}
	archive := filepath.Join(root, "engine.zip")
	if err := os.WriteFile(archive, []byte("unreviewed archive"), 0o600); err != nil {
		t.Fatal(err)
	}
	engine := filepath.Join(root, "engine")
	err = run(t.Context(), []string{"engine", "install", "-engine-dir", engine,
		"-state", filepath.Join(root, "state"), "-engine-zip", archive})
	if err == nil || !strings.Contains(err.Error(), "ZIP") {
		t.Fatalf("unpinned archive accepted or serve invoked: %v", err)
	}
	if _, err := os.Stat(engine); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("refused archive changed engine path: %v", err)
	}
}
