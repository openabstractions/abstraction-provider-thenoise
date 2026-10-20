package thenoise

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

// testRuntime is a Runtime over the miniature engine with fake gate and cache
// answers; reported collects OnError reports.
func testRuntime(t *testing.T, device string) (*Runtime, *[]error) {
	t.Helper()
	engineDir, stateDir, pin := installSmall(t)
	var mu sync.Mutex
	reported := &[]error{}
	r := &Runtime{
		cfg: ManagedConfig{Launch: LaunchConfig{RuntimeDir: engineDir, StateDir: stateDir, Device: device},
			OnError: func(err error) { mu.Lock(); *reported = append(*reported, err); mu.Unlock() }},
		probe: fakeProbe(), cacheSec: newFakeSecurity().sec(), manifestPin: pin, files: testFiles(),
	}
	return r, reported
}

// pre runs the launch preflight and releases its hold, as for an engine that
// exits at once.
func pre(t *testing.T, r *Runtime) (string, []string, error) {
	t.Helper()
	device, env, hold, err := r.preflight(t.Context())
	if err := hold.Release(); err != nil {
		t.Fatal(err)
	}
	return device, env, err
}

// deepen makes the miniature engine's hipBLASLt path one character too long.
// The manifest pin follows the change, so only the gate objects.
func deepen(t *testing.T, r *Runtime) {
	t.Helper()
	hip := gpuLibraries[0]
	gfx := libraryDir(r.cfg.Launch.RuntimeDir, hip.name)
	name := nameForTotal(t, gfx, 1, maxLibraryPath+1)
	if err := os.WriteFile(filepath.Join(gfx, name), []byte("k"), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := readManifest(manifestFile(r.cfg.Launch.StateDir), r.manifestPin)
	if err != nil {
		t.Fatal(err)
	}
	rel := filepath.ToSlash(strings.TrimPrefix(filepath.Join(gfx, name), r.cfg.Launch.RuntimeDir+string(filepath.Separator)))
	m.entries = append(m.entries, manifestEntry{Path: rel, SHA256: digestHex([]byte("k")), Size: 1})
	encoded := m.encode()
	if err := os.WriteFile(manifestFile(r.cfg.Launch.StateDir), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	r.manifestPin = digestHex(encoded)
}

func TestPreflightChoosesTheGPUOnlyWhenTheGatePasses(t *testing.T) {
	r, _ := testRuntime(t, "auto")
	device, env, err := pre(t, r)
	if err != nil || device != "cuda" {
		t.Fatalf("auto with a passing gate chose %q: %v", device, err)
	}
	if got := envMap(env); got["HF_HUB_OFFLINE"] != "1" || got["AMD_COMGR_CACHE_DIR"] == "" {
		t.Fatalf("GPU launch environment %v", got)
	}
	deepen(t, r)
	device, env, err = pre(t, r)
	if err != nil || device != "cpu" {
		t.Fatalf("auto with a failing gate chose %q: %v", device, err)
	}
	if _, ok := envMap(env)["HIPBLASLT_TENSILE_LIBPATH"]; ok {
		t.Fatal("a CPU launch carried a TENSILE_LIBPATH variable")
	}
}

func TestPreflightRefusesCUDAWhenTheGateFails(t *testing.T) {
	r, _ := testRuntime(t, "cuda")
	deepen(t, r)
	_, _, err := pre(t, r)
	if !errors.Is(err, ErrGPUPathTooDeep) {
		t.Fatalf("cuda launch past a failing gate: %v", err)
	}
	if !strings.Contains(err.Error(), r.cfg.Launch.RuntimeDir) {
		t.Fatalf("the refusal does not name the engine folder: %v", err)
	}
}

func TestPreflightReportsWhyAutoStaysOnTheCPU(t *testing.T) {
	r, reported := testRuntime(t, "auto")
	deepen(t, r)
	if _, _, err := pre(t, r); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, err := range *reported {
		found = found || errors.Is(err, ErrGPUPathTooDeep) && strings.Contains(err.Error(), r.cfg.Launch.RuntimeDir)
	}
	if !found {
		t.Fatalf("no report of the gate refusal naming the engine folder: %v", *reported)
	}
}

func TestPreflightVerifiesTheTreeBeforeEveryLaunch(t *testing.T) {
	r, _ := testRuntime(t, "cpu")
	if _, _, err := pre(t, r); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(r.cfg.Launch.RuntimeDir, "thenoise.bat"), []byte("@echo evil\r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := pre(t, r); err == nil {
		t.Fatal("a changed launcher passed the preflight")
	}
}

func TestAccessViolationOfAGPUGenerationTurnsTheGPUOff(t *testing.T) {
	r, reported := testRuntime(t, "auto")
	const violation = int(0xC0000005)
	(&engineProcess{owner: r, gpu: false}).observeExit(violation)
	(&engineProcess{owner: r, gpu: true}).observeExit(1) // stopped by the provider
	if r.gpuOff.Load() {
		t.Fatal("a CPU crash or a provider stop turned the GPU off")
	}
	(&engineProcess{owner: r, gpu: true}).observeExit(violation)
	if !r.gpuOff.Load() {
		t.Fatal("a GPU access violation left the GPU on")
	}
	found := false
	for _, err := range *reported {
		found = found || errors.Is(err, ErrGPUCrashed)
	}
	if !found {
		t.Fatal("the GPU crash was not reported")
	}
	if device, _, err := pre(t, r); err != nil || device != "cpu" {
		t.Fatalf("auto relaunched on the GPU after a crash: %q %v", device, err)
	}
	r.cfg.Launch.Device = "cuda"
	if _, _, err := pre(t, r); !errors.Is(err, ErrGPUCrashed) {
		t.Fatalf("cuda relaunched after a crash: %v", err)
	}
}

func TestPreflightKeepsTheCachesPrivate(t *testing.T) {
	r, reported := testRuntime(t, "cpu")
	f := newFakeSecurity()
	r.cacheSec = f.sec()
	if _, _, err := pre(t, r); err != nil {
		t.Fatal(err)
	}
	present := plant(t, r.cfg.Launch.StateDir)
	f.public[filepath.Join(cacheRoot(r.cfg.Launch.StateDir), "comgr")] = true
	if _, _, err := pre(t, r); err != nil {
		t.Fatal(err)
	}
	if present() || len(*reported) == 0 {
		t.Fatalf("a cache behind a changed ACL survived the preflight: %v", *reported)
	}
}

func TestRunNamesTheGPUPathRefusal(t *testing.T) {
	e := managedEngine(Config{Endpoint: "http://127.0.0.1:1", Model: "flux_klein", MaxRun: time.Minute},
		func(context.Context) (func(), *http.Client, func() error, error) {
			return nil, nil, nil, errors.Join(errors.New("holder: start"), ErrGPUPathTooDeep)
		})
	got := e.Run(t.Context(), "flux_klein", requestFor(wire.ImageModeGenerate), nil, nil)
	if got.Outcome != wire.ReplyOutcomeUnavailable || got.Reason != "engine:gpu-path-too-deep" {
		t.Fatalf("reply %v %q", got.Outcome, got.Reason)
	}
}

func TestRunNamesTheGPUCrashRefusal(t *testing.T) {
	e := managedEngine(Config{Endpoint: "http://127.0.0.1:1", Model: "flux_klein", MaxRun: time.Minute},
		func(context.Context) (func(), *http.Client, func() error, error) {
			return nil, nil, nil, errors.Join(errors.New("holder: start"), ErrGPUCrashed)
		})
	got := e.Run(t.Context(), "flux_klein", requestFor(wire.ImageModeGenerate), nil, nil)
	if got.Outcome != wire.ReplyOutcomeUnavailable || got.Reason != "engine:gpu-crashed" {
		t.Fatalf("reply %v %q", got.Outcome, got.Reason)
	}
}
