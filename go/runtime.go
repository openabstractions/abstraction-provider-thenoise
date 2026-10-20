package thenoise

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-facade/go/supervision"
	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-provider-kit/go/enginehttp"
	"github.com/openabstractions/abstraction-provider-kit/go/enginelaunch"
	"github.com/openabstractions/abstraction-resource/go/holder"
	router "github.com/openabstractions/abstraction-router/go"
)

// EngineZIPSHA256 pins the one reviewed TheNoise artifact,
// thenoise-0.9.1-rocm10.1.0-gfx1151-win-x64.zip of lemonade-sdk/thenoise
// release thenoise-0.9.1-rocm10.1.0 (research/security-reviews/
// thenoise-0.9.1-rocm10.1.0-gfx1151-win-x64.md, condition 1). The provider
// verifies it before reading any entry of the ZIP.
const EngineZIPSHA256 = "0a639633d6aff9523190c557f2377ba8ef4afab5b02644b3778082793cbdf737"

// EngineZIPBytes is that artifact's size, recorded by the same review.
const EngineZIPBytes int64 = 1633390238

// EngineManifestSHA256 pins the manifest recorded from that ZIP: the path,
// SHA-256 and size of each of its files. The extracted tree is verified
// against the manifest before every launch.
const EngineManifestSHA256 = "fd011267b18703b8092fe283a299824d57d9d63b5c208f56397a0886111339aa"

// EngineFolderName is the folder the provider extracts the pinned engine into
// below %LOCALAPPDATA%\OA\engines.
const EngineFolderName = "thenoise-0.9.1"

// DefaultEngineDir is %LOCALAPPDATA%\OA\engines\thenoise-0.9.1. Under C:\Users
// it is 49 characters plus the profile folder name, within MaxEngineDirLength
// for profile folder names of up to 38 characters.
func DefaultEngineDir() (string, error) {
	local := os.Getenv("LOCALAPPDATA")
	if local == "" || !filepath.IsAbs(local) {
		return "", errors.New("thenoise: LOCALAPPDATA is required for the default engine folder")
	}
	return filepath.Join(local, "OA", "engines", EngineFolderName), nil
}

// manifestFile is where the engine manifest lives in the provider state.
func manifestFile(stateDir string) string {
	return filepath.Join(stateDir, EngineFolderName+".manifest")
}

// LaunchConfig names the portable TheNoise installation, the provider's
// private state folder and the three model files. TheNoise's official
// thenoise.bat must be used on Windows so its bundled ROCm DLLs and supported
// environment are set up. EngineZIP, when set, is the pinned ZIP the provider
// installs from when no manifest has been recorded yet.
type LaunchConfig struct {
	RuntimeDir     string
	StateDir       string
	EngineZIP      string
	DiT            string
	VAE            string
	TextEncoder    string
	Device         string
	Port           int
	ReadyTimeout   time.Duration
	ForceTimeout   time.Duration
	Stdout, Stderr *os.File
}

// ManagedConfig combines one local image model, its explicit resource lease,
// and the supported TheNoise launcher paths.
type ManagedConfig struct {
	Engine   Config
	Launch   LaunchConfig
	Leases   holder.Leases
	Resource string
	Amount   int64
	Idle     time.Duration
	Renew    time.Duration
	Wait     time.Duration
	OnError  func(error)
}

// Runtime owns TheNoise process generations and their resource leases.
type Runtime struct {
	cfg     ManagedConfig
	holder  *holder.Holder
	engine  *Engine
	mu      sync.RWMutex
	current *engineProcess
	active  func(string) bool
	closed  atomic.Bool
	// probe, cacheSec and manifestPin are the gate's system questions, the
	// cache guard's account and ACL operations and the manifest pin; tests
	// replace them.
	probe       gpuProbe
	cacheSec    cacheSecurity
	manifestPin string
	// files proves and holds the engine tree for this provider process
	// (INF-F1..INF-F4); tests give it a fake folder ACL.
	files *programfiles.Verifier
	// gpuOff latches after a GPU generation exits with an access violation
	// (security review condition 8); it clears only with a new provider.
	gpuOff atomic.Bool
	// sockets reads the socket owner tables for the loopback check; nil
	// reads them through abstraction-identity. Tests replace it.
	sockets func() ([]identity.Socket, error)
}

type engineProcess struct {
	proc *supervision.Process
	// contains answers whether a process belongs to this generation's
	// contained tree: the supervised job's membership.
	contains identity.TreeContains
	client   *http.Client
	owner    *Runtime
	gpu      bool
	alive    atomic.Bool
	once     sync.Once
	exited   sync.Once
	// hold keeps the engine tree's files open with read sharing only until
	// the process tree's exit is confirmed.
	hold *programfiles.Hold
}

// released closes the held engine files once the process tree has exited.
// An unconfirmed exit keeps them held.
func (p *engineProcess) released(waitErr error) {
	if waitErr == nil {
		p.hold.Release()
	}
}

// statusAccessViolation is the exit code of a process killed by an
// unhandled access violation, as cmd.exe passes python.exe's on.
const statusAccessViolation = 0xC0000005

// ErrGPUCrashed refuses a --device cuda launch after a GPU generation of this
// provider crashed with an access violation.
var ErrGPUCrashed = errors.New("TheNoise crashed on the GPU; GPU launches stay off until the provider restarts.")

// observeExit records how a generation ended. An access violation of a GPU
// generation turns GPU launches off for the life of the provider: the gate
// passed, and a second path fault would crash again.
func (p *engineProcess) observeExit(code int) {
	p.exited.Do(func() {
		if p.gpu && uint32(code) == statusAccessViolation {
			if p.owner.gpuOff.CompareAndSwap(false, true) && p.owner.cfg.OnError != nil {
				p.owner.cfg.OnError(ErrGPUCrashed)
			}
		}
	})
}

// NewRuntime prepares the image callback and lazy resource owner. It starts no
// process; an accepted operation acquires the lease and runs the launcher.
func NewRuntime(cfg ManagedConfig) (*Runtime, error) {
	if cfg.Engine.Model == "" || cfg.Engine.MaxRun <= 0 || cfg.Leases == nil || cfg.Resource == "" || cfg.Amount <= 0 {
		return nil, errors.New("thenoise: model, max run, resource lease and positive amount are required")
	}
	if err := validateLaunch(cfg.Launch); err != nil {
		return nil, err
	}
	if _, err := commandLine(cfg.Launch, cfg.Engine.Endpoint); err != nil {
		return nil, err
	}
	r := &Runtime{cfg: cfg, probe: systemGPUProbe(), cacheSec: systemCacheSecurity(), manifestPin: EngineManifestSHA256,
		files: programfiles.NewVerifier(programfiles.Options{})}
	h, err := holder.New(holder.Options{Leases: cfg.Leases, Resource: cfg.Resource, Idle: cfg.Idle, Renew: cfg.Renew, Wait: cfg.Wait, OnError: cfg.OnError,
		Active: func(name string) bool {
			r.mu.RLock()
			active := r.active
			r.mu.RUnlock()
			return active != nil && active(name)
		}})
	if err != nil {
		return nil, err
	}
	r.holder = h
	r.engine = managedEngine(cfg.Engine, r.acquire)
	return r, nil
}

// Engine returns the provider callback supplied to providerhost.NewImage.
func (r *Runtime) Engine() *Engine { return r.engine }

// Admit checks the configured single model without loading it. Slow resource
// acquisition and readiness happen inside the accepted operation, where they
// remain cancellable and visible through operation progress. Another model
// reads model:undeclared (45bfa74923). Rights owns not_permitted, which
// fails a durable job when returned by its hook (inference INF-A5).
// A closing runtime reads unavailable. A job recovered by the next process asks again.
func (r *Runtime) Admit(ctx context.Context, _ *router.Host, model string) (string, bool) {
	if ctx.Err() != nil {
		return "caller", false
	}
	if model != r.cfg.Engine.Model {
		return "model:undeclared", false
	}
	if r.closed.Load() {
		return "unavailable", false
	}
	return "", true
}

// BindActivity connects the lease holder's yield guard to the inference
// provider's pending, executing and retained operation activity.
func (r *Runtime) BindActivity(active func(string) bool) error {
	if active == nil {
		return errors.New("thenoise: activity callback required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != nil {
		return errors.New("thenoise: activity callback already bound")
	}
	r.active = active
	return nil
}

func (r *Runtime) ensure(ctx context.Context) error {
	return r.holder.Ensure(ctx, r.cfg.Engine.Model, func() (holder.Component, error) {
		return holder.Component{Amount: r.cfg.Amount, Start: func() (holder.Owned, error) {
			owned, err := r.start(ctx)
			if err != nil {
				return nil, err
			}
			r.mu.Lock()
			r.current = owned
			r.mu.Unlock()
			return owned, nil
		}}, nil
	})
}

func (r *Runtime) acquire(ctx context.Context) (func(), *http.Client, func() error, error) {
	if r.closed.Load() {
		return nil, nil, nil, errors.New("provider closing")
	}
	end, err := r.holder.Begin(ctx, r.cfg.Engine.Model)
	if err != nil {
		return nil, nil, nil, err
	}
	if err := r.ensure(ctx); err != nil {
		end()
		return nil, nil, nil, err
	}
	r.mu.RLock()
	current := r.current
	r.mu.RUnlock()
	if current == nil || !current.Alive() {
		end()
		return nil, nil, nil, errors.New("TheNoise generation unavailable")
	}
	return end, current.client, current.stopAndWait, nil
}

func (r *Runtime) start(ctx context.Context) (*engineProcess, error) {
	device, env, hold, err := r.preflight(ctx)
	if err != nil {
		if r.cfg.OnError != nil {
			r.cfg.OnError(err)
		}
		return nil, err
	}
	for _, path := range []string{filepath.Join(r.cfg.Launch.RuntimeDir, "thenoise.bat"), r.cfg.Launch.DiT, r.cfg.Launch.VAE, r.cfg.Launch.TextEncoder} {
		if _, err := os.Stat(path); err != nil {
			hold.Release()
			return nil, fmt.Errorf("thenoise: required launcher/model path: %w", err)
		}
	}
	// cmd.exe is found in the kernel's Windows folder, the SystemRoot the
	// clean environment carries, never through the provider's environment.
	systemRoot := ""
	for _, kv := range env {
		if name, value, _ := strings.Cut(kv, "="); strings.EqualFold(name, "SystemRoot") {
			systemRoot = value
		}
	}
	if !filepath.IsAbs(systemRoot) {
		hold.Release()
		return nil, errors.New("thenoise: the engine environment has no Windows folder")
	}
	cmd := filepath.Join(systemRoot, "System32", "cmd.exe")
	launch := r.cfg.Launch
	launch.Device = device
	command, err := commandLine(launch, r.cfg.Engine.Endpoint)
	if err != nil {
		hold.Release()
		return nil, err
	}
	stdout, stderr := r.cfg.Launch.Stdout, r.cfg.Launch.Stderr
	if stdout == nil {
		stdout = os.Stdout
	}
	if stderr == nil {
		stderr = os.Stderr
	}
	process, err := supervision.Start(ctx, supervision.Options{Executable: cmd, WindowsCmdLine: fmt.Sprintf(`"%s" /d /s /c %s`, cmd, command), Dir: r.cfg.Launch.RuntimeDir,
		Env: env, Stdin: os.Stdin, Stdout: stdout, Stderr: stderr, ForceTimeout: r.cfg.Launch.ForceTimeout})
	if err != nil {
		hold.Release()
		return nil, err
	}
	current := &engineProcess{proc: process, contains: process.Contains, owner: r, gpu: device == "cuda", hold: hold}
	logTransport(stderr)
	current.alive.Store(true)
	current.client, err = engineClient(r.cfg.Engine.Endpoint, process)
	if err != nil {
		_ = process.Stop()
		code, waitErr := process.Wait()
		current.observeExit(code)
		current.released(waitErr)
		current.alive.Store(false)
		return nil, err
	}
	go func() {
		code, waitErr := process.Wait()
		current.observeExit(code)
		current.released(waitErr)
		current.alive.Store(false)
		if waitErr != nil && r.cfg.OnError != nil {
			r.cfg.OnError(waitErr)
		}
		current.clear()
	}()
	return current, nil
}

func logTransport(output io.Writer) {
	const reason = "IPC not implemented in this adopter"
	slog.Info("engine transport", "engine", "TheNoise", "pin", EngineZIPSHA256, "transport", "tcp-loopback", "reason", reason)
	if output != nil {
		//unchecked: informational diagnostics do not alter the admitted launch
		fmt.Fprintf(output, "INFO engine=TheNoise pin=%s transport=tcp-loopback reason=%q\n", EngineZIPSHA256, reason)
	}
}

// preflight runs before every launch. It installs the engine from the pinned
// ZIP when no manifest exists and a ZIP is configured, proves the tree
// against the pinned manifest and holds it (INF-F1..INF-F4), chooses the
// device, keeps the ROCm cache folders private and returns the device, the
// complete engine environment and the hold the launch keeps until the engine
// exits.
//
// Device choice: cpu stays cpu. cuda runs on the GPU only when the readiness
// gate passes and no GPU generation has crashed; otherwise the launch is
// refused. auto runs on the GPU under the same two conditions and on the CPU
// otherwise, reporting why.
func (r *Runtime) preflight(ctx context.Context) (string, []string, *programfiles.Hold, error) {
	launch := r.cfg.Launch
	manifest := manifestFile(launch.StateDir)
	if launch.EngineZIP != "" {
		if _, err := os.Stat(manifest); errors.Is(err, fs.ErrNotExist) {
			if err := installEngine(ctx, r.files, launch.EngineZIP, EngineZIPSHA256, EngineZIPBytes, launch.RuntimeDir, manifest, r.manifestPin); err != nil {
				return "", nil, nil, err
			}
		}
	}
	hold, err := launchEngine(ctx, r.files, launch.RuntimeDir, manifest, r.manifestPin)
	if err != nil {
		return "", nil, nil, err
	}
	device, env, err := r.chooseDevice()
	if err != nil {
		hold.Release()
		return "", nil, nil, err
	}
	return device, env, hold, nil
}

// chooseDevice applies the GPU gate and the cache guard to a proven tree.
func (r *Runtime) chooseDevice() (string, []string, error) {
	launch := r.cfg.Launch
	device := "cpu"
	var gpu []string
	if launch.Device == "cuda" || launch.Device == "auto" {
		var err error
		if r.gpuOff.Load() {
			err = ErrGPUCrashed
		} else {
			gpu, err = gpuGate(launch.RuntimeDir, r.probe)
		}
		switch {
		case err == nil:
			device = "cuda"
		case launch.Device == "cuda":
			return "", nil, fmt.Errorf("thenoise: engine folder %s: %w", launch.RuntimeDir, err)
		default:
			gpu = nil
			if r.cfg.OnError != nil {
				r.cfg.OnError(fmt.Errorf("thenoise: running on the CPU; engine folder %s: %w", launch.RuntimeDir, err))
			}
		}
	}
	reason, err := prepareCaches(launch.StateDir, r.manifestPin, r.cacheSec)
	if err != nil {
		return "", nil, err
	}
	if reason != "" && reason != cachesNew && r.cfg.OnError != nil {
		r.cfg.OnError(fmt.Errorf("thenoise: ROCm caches emptied: %s", reason))
	}
	env, err := launchEnv(launch.StateDir, gpu, r.cfg.OnError)
	if err != nil {
		return "", nil, fmt.Errorf("thenoise: engine environment: %w", err)
	}
	return device, env, nil
}

func engineClient(endpoint string, process *supervision.Process) (*http.Client, error) {
	return enginehttp.NewClientWithVerifier(endpoint, func(peer identity.Process) (bool, error) {
		return process.Contains(peer.PID, peer.StartTime)
	})
}

func (p *engineProcess) PID() int    { return p.proc.PID() }
func (p *engineProcess) Alive() bool { return p.alive.Load() }
func (p *engineProcess) clear() {
	p.once.Do(func() {
		p.owner.mu.Lock()
		if p.owner.current == p {
			p.owner.current = nil
		}
		p.owner.mu.Unlock()
	})
}
func (p *engineProcess) stopAndWait() error {
	_ = p.proc.Stop()
	code, err := p.proc.Wait()
	p.observeExit(code)
	p.released(err)
	if err == nil {
		p.alive.Store(false)
		p.clear()
	}
	return err
}
func (p *engineProcess) Stop() bool { return p.stopAndWait() == nil }

func (p *engineProcess) Ready(ctx context.Context) error {
	deadline := time.Now().Add(p.owner.cfg.Launch.ReadyTimeout)
	// A connect that can never succeed ends readiness after
	// enginehttp.UnreachableFor, the rule every engine readiness keeps.
	var run enginehttp.UnreachableRun
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !p.Alive() {
			return errors.New("thenoise: owned engine exited before readiness")
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.owner.cfg.Engine.Endpoint+"/health", nil)
		if err == nil {
			resp, doErr := p.client.Do(req)
			if end := run.Note(doErr); end != nil {
				return fmt.Errorf("thenoise: readiness: %w", end)
			}
			if doErr == nil {
				data, readErr := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
				_ = resp.Body.Close()
				if readErr == nil && resp.StatusCode == http.StatusOK {
					var health struct {
						Status string   `json:"status"`
						Models []string `json:"models"`
					}
					if json.Unmarshal(data, &health) == nil && health.Status == "ok" && slices.Contains(health.Models, p.owner.cfg.Engine.Model) {
						return p.loopbackOnly()
					}
				}
			}
		}
		if !time.Now().Before(deadline) {
			return errors.New("thenoise: readiness timeout or configured model absent from /health")
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}

// loopbackOnly is the provider kit's loopback check (inference CONTRACT.md
// INF-EL2) of the generation's contained tree, once it reports ready:
// thenoise.bat serve opens its listener before /health answers, and the
// generation starts no listener later. A refusal wraps abstraction-identity's
// ErrBeyondLoopback (ID-T7); it is a failed readiness, and the holder stops
// the tree and releases its lease.
func (p *engineProcess) loopbackOnly() error {
	if err := enginelaunch.CheckLoopback(p.owner.sockets, p.owner.cfg.Launch.Port, p.contains); err != nil {
		return fmt.Errorf("thenoise: %w", err)
	}
	return nil
}

func (r *Runtime) Run(ctx context.Context) { r.holder.Run(ctx) }

// Close releases all held resource leases only after their process trees stop.
func (r *Runtime) Close() {
	if r.closed.CompareAndSwap(false, true) {
		r.holder.Close()
	}
}

func validateLaunch(cfg LaunchConfig) error {
	for label, path := range map[string]string{"runtime directory": cfg.RuntimeDir, "state directory": cfg.StateDir, "DiT": cfg.DiT, "VAE": cfg.VAE, "text encoder": cfg.TextEncoder} {
		if !filepath.IsAbs(path) {
			return fmt.Errorf("thenoise: %s must be an absolute path", label)
		}
	}
	if cfg.EngineZIP != "" && !filepath.IsAbs(cfg.EngineZIP) {
		return errors.New("thenoise: engine ZIP must be an absolute path")
	}
	switch cfg.Device {
	case "", "cpu", "cuda", "auto":
	default:
		return errors.New("thenoise: device must be auto, cpu or cuda (PyTorch ROCm device word)")
	}
	if cfg.Port < 1 || cfg.Port > 65535 {
		return errors.New("thenoise: engine port must be 1..65535")
	}
	if cfg.ReadyTimeout <= 0 || cfg.ForceTimeout <= 0 {
		return errors.New("thenoise: readiness and process-exit timeouts are required")
	}
	for _, path := range []string{cfg.RuntimeDir, cfg.StateDir, cfg.EngineZIP, cfg.DiT, cfg.VAE, cfg.TextEncoder} {
		if strings.ContainsAny(path, `"%!&|<>^`) {
			return errors.New("thenoise: configured paths contain unsupported cmd.exe metacharacters")
		}
	}
	return nil
}

func commandLine(cfg LaunchConfig, endpoint string) (string, error) {
	if err := validateLaunch(cfg); err != nil {
		return "", err
	}
	if !strings.HasPrefix(endpoint, "http://127.0.0.1:") {
		return "", errors.New("thenoise: engine endpoint must be local IPv4 loopback")
	}
	if _, err := strconv.Atoi(strings.TrimPrefix(endpoint, "http://127.0.0.1:")); err != nil {
		return "", errors.New("thenoise: invalid engine endpoint port")
	}
	batch := filepath.Join(cfg.RuntimeDir, "thenoise.bat")
	port := cfg.Port
	if endpointPort := strings.TrimPrefix(endpoint, "http://127.0.0.1:"); endpointPort != strconv.Itoa(port) {
		return "", errors.New("thenoise: launch port and engine endpoint differ")
	}
	// A launch names the device the preflight chose; auto is never passed on.
	device := cfg.Device
	if device != "cuda" {
		device = "cpu"
	}
	// cmd /S /C removes this outer quote pair and invokes only the official
	// launcher. Configured values are path-validated before interpolation.
	return fmt.Sprintf(`""%s" serve --dit "%s" --vae "%s" --text-encoder "%s" --host 127.0.0.1 --port %d --device %s"`, batch, cfg.DiT, cfg.VAE, cfg.TextEncoder, port, device), nil
}
