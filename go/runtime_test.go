package thenoise

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	resourceclient "github.com/openabstractions/abstraction-resource/go/client"
)

func TestOfficialLauncherCommandQuotesPathsAndSelectsCPU(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("official launcher is Windows only: thenoise.bat under cmd.exe, from the pinned win-x64 artifact")
	}
	launch := LaunchConfig{RuntimeDir: `C:\Program Files\TheNoise`, StateDir: `C:\State\thenoise`, DiT: `C:\Models\flux klein.safetensors`, VAE: `C:\Models\vae.safetensors`,
		TextEncoder: `C:\Models\qwen.safetensors`, Port: 18118, ReadyTimeout: time.Minute, ForceTimeout: 10 * time.Second}
	got, err := commandLine(launch, "http://127.0.0.1:18118")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`""C:\Program Files\TheNoise\thenoise.bat" serve`, `--dit "C:\Models\flux klein.safetensors"`, `--vae "C:\Models\vae.safetensors"`, `--text-encoder "C:\Models\qwen.safetensors"`, `--device cpu"`} {
		if !strings.Contains(got, want) {
			t.Fatalf("official launcher argument %q missing from %q", want, got)
		}
	}
}

func TestLauncherRejectsCommandMetacharacters(t *testing.T) {
	launch := LaunchConfig{RuntimeDir: `C:\Program Files\TheNoise`, StateDir: `C:\State\thenoise`, DiT: `C:\Models\dit.safetensors`, VAE: `C:\Models\vae.safetensors`,
		TextEncoder: `C:\Models\qwen&bad.safetensors`, Port: 18118, ReadyTimeout: time.Minute, ForceTimeout: 10 * time.Second}
	if _, err := commandLine(launch, "http://127.0.0.1:18118"); err == nil {
		t.Fatal("cmd.exe metacharacter was accepted")
	}
}

func TestReadyRequiresActualHealthStatusAndModels(t *testing.T) {
	// Exact health response captured from the pinned 0.9.1 engine after loading.
	response := `{"status":"ok","models":["flux_klein"],"capabilities":{"edit":true,"kv_cache":true},"pixel_channels":3}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("unexpected health path %s", r.URL.Path)
		}
		_, _ = fmt.Fprint(w, response)
	}))
	defer server.Close()
	_, process := loopbackEngine(t, server)
	if err := process.Ready(context.Background()); !loopbackAnswer(err) {
		t.Fatal(err)
	}
	response = `{"status":"ok","models":["different","flux_klein"]}`
	if err := process.Ready(context.Background()); !loopbackAnswer(err) {
		t.Fatal("configured model in list failed readiness:", err)
	}
	for _, invalid := range []string{
		`{"status":"ok","models":["different"]}`,
		`{"status":"ok","models":[]}`,
		`{"status":"ok","model":"flux_klein"}`,
		`{"status":"loading","models":["flux_klein"]}`,
		`{"status":"ok","models":"flux_klein"}`,
	} {
		response = invalid
		if err := process.Ready(context.Background()); err == nil {
			t.Fatalf("invalid health passed readiness: %s", response)
		}
	}
}

// loopbackEngine is a ready-checked generation whose health server stands
// at server and whose contained tree is this test process, listening on
// server's loopback port in the owner table its Runtime reads.
func loopbackEngine(t *testing.T, server *httptest.Server, extra ...identity.Socket) (*Runtime, *engineProcess) {
	t.Helper()
	addr := netip.MustParseAddrPort(strings.TrimPrefix(server.URL, "http://"))
	owner := &Runtime{cfg: ManagedConfig{Engine: Config{Endpoint: server.URL, Model: "flux_klein"},
		Launch: LaunchConfig{ReadyTimeout: 100 * time.Millisecond, Port: int(addr.Port())}}}
	owner.sockets = func() ([]identity.Socket, error) {
		return append([]identity.Socket{{Proto: "tcp", Addr: addr.Addr(), Port: int(addr.Port()), PID: os.Getpid()}}, extra...), nil
	}
	process := &engineProcess{client: server.Client(), owner: owner,
		contains: func(pid int, _ time.Time) (bool, error) { return pid == os.Getpid(), nil }}
	process.alive.Store(true)
	return owner, process
}

// loopbackAnswer reports a readiness that passed, or on a platform without
// socket owner tables or process creation times (outside Windows) the
// loopback check's refusal, which refuses every launch there.
func loopbackAnswer(err error) bool {
	if runtime.GOOS == "windows" {
		return err == nil
	}
	return errors.Is(err, identity.ErrBeyondLoopback)
}

// Inference CONTRACT.md INF-EL2: once /health answers, the generation's
// contained tree must hold its listener on 127.0.0.1 at the launch port and
// no socket beyond loopback. A tree that also listens on 0.0.0.0 fails
// readiness, and the holder stops it.
func TestReadyRefusesAListenerBeyondLoopback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprint(w, `{"status":"ok","models":["flux_klein"]}`)
	}))
	defer server.Close()
	addr := netip.MustParseAddrPort(strings.TrimPrefix(server.URL, "http://"))
	_, process := loopbackEngine(t, server, identity.Socket{Proto: "tcp", Addr: netip.MustParseAddr("0.0.0.0"), Port: int(addr.Port()) + 1, PID: os.Getpid()})
	if err := process.Ready(context.Background()); !errors.Is(err, identity.ErrBeyondLoopback) {
		t.Fatalf("a tree listening beyond loopback passed readiness: %v", err)
	}
	_, process = loopbackEngine(t, server)
	process.owner.cfg.Launch.Port = int(addr.Port()) + 1
	if err := process.Ready(context.Background()); !errors.Is(err, identity.ErrBeyondLoopback) {
		t.Fatalf("a tree without its listener at the launch port passed readiness: %v", err)
	}
	_, process = loopbackEngine(t, server)
	process.owner.sockets = func() ([]identity.Socket, error) { return nil, errors.New("no owner table") }
	if err := process.Ready(context.Background()); !errors.Is(err, identity.ErrBeyondLoopback) {
		t.Fatalf("an unreadable owner table passed readiness: %v", err)
	}
}

func TestManagedHostDeclaresExactModelWithoutStartingEngine(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("official launcher is Windows only")
	}
	dir := t.TempDir()
	launch := LaunchConfig{RuntimeDir: dir, StateDir: filepath.Join(dir, "state"), DiT: filepath.Join(dir, "dit.safetensors"), VAE: filepath.Join(dir, "vae.safetensors"),
		TextEncoder: filepath.Join(dir, "text.safetensors"), Device: "cpu", Port: 18118,
		ReadyTimeout: time.Minute, ForceTimeout: time.Second}
	managed, err := NewRuntime(ManagedConfig{Engine: Config{Endpoint: "http://127.0.0.1:18118", Model: "flux_klein", MaxRun: time.Minute},
		Launch: launch, Leases: resourceclient.Discover(), Resource: "memory", Amount: 1, Wait: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	defer managed.Close()
	if reason, ok := managed.Admit(context.Background(), nil, "flux_klein"); !ok || reason != "" {
		t.Fatalf("valid model refused: %q", reason)
	}
	if managed.current != nil {
		t.Fatal("admission launched engine")
	}
	if reason, ok := managed.Admit(context.Background(), nil, "other"); ok || reason != "not_permitted" {
		t.Fatalf("wrong model admitted: %q", reason)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	endpoint := fmt.Sprintf(`\\.\pipe\thenoise-host-test-%d-%d`, os.Getpid(), time.Now().UnixNano())
	host, err := NewManagedHost(HostConfig{Endpoint: endpoint, RuntimePrograms: []string{exe}, StoreRoot: filepath.Join(dir, "store"),
		RecordPath: filepath.Join(dir, "records.json"), HostName: "thenoise-fixture"}, managed)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	if managed.active == nil || managed.current != nil {
		t.Fatal("host activity was not wired or engine started early")
	}
}
