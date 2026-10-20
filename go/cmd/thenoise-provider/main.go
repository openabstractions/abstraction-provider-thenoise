// Command thenoise-provider serves one local TheNoise image model through the
// standard OA inference and content endpoint.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	thenoise "github.com/openabstractions/abstraction-provider-thenoise/go"
	resourceclient "github.com/openabstractions/abstraction-resource/go/client"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string) error {
	if len(args) > 0 && args[0] == "engine" {
		return runEngine(ctx, args[1:], thenoise.InstallEngine, thenoise.UnlockEngine)
	}
	flags := flag.NewFlagSet("thenoise-provider", flag.ContinueOnError)
	var endpoint, runtimeProgram, storeRoot, recordPath, auditPath, hostName string
	var engineDir, engineZIP, stateDir, dit, vae, textEncoder, model, device, resource, resourceEndpoint string
	var port int
	var amount int64
	flags.StringVar(&endpoint, "endpoint", "", "OA native provider endpoint")
	flags.StringVar(&runtimeProgram, "runtime", "", "absolute path of the designated OA runtime executable")
	flags.StringVar(&storeRoot, "store", "", "private provider content directory")
	flags.StringVar(&recordPath, "records", "", "private provider content record file")
	flags.StringVar(&auditPath, "audit", "", "absolute provider audit journal (default: beside -records, <name>-audit.jsonl)")
	flags.StringVar(&hostName, "host", "thenoise-local", "declared provider host name")
	flags.StringVar(&engineDir, "engine-dir", "", `folder holding official thenoise.bat (default: %LOCALAPPDATA%\OA\engines\thenoise-0.9.1)`)
	flags.StringVar(&engineZIP, "engine-zip", "", "absolute path of the pinned TheNoise ZIP; installs the engine when no manifest is recorded")
	flags.StringVar(&stateDir, "state", "", "absolute private state folder for the engine manifest and ROCm caches (default: beside -records, thenoise-state)")
	flags.StringVar(&dit, "dit", "", "absolute BF16 DiT model path")
	flags.StringVar(&vae, "vae", "", "absolute VAE model path")
	flags.StringVar(&textEncoder, "text-encoder", "", "absolute text encoder model path")
	flags.StringVar(&model, "model", "flux_klein", "exact model ID reported by TheNoise /health")
	flags.StringVar(&device, "device", "auto", "TheNoise device: auto (GPU when its readiness gate passes, else CPU), cpu or cuda")
	flags.StringVar(&resource, "resource", "memory", "OA resource lease name")
	flags.StringVar(&resourceEndpoint, "resource-endpoint", "", "OA resource service endpoint (default: local discovery)")
	flags.IntVar(&port, "port", 18118, "private 127.0.0.1 engine port")
	flags.Int64Var(&amount, "amount-bytes", 0, "required resource reservation amount in bytes")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("thenoise-provider: unexpected positional arguments")
	}
	if engineDir == "" {
		dir, err := thenoise.DefaultEngineDir()
		if err != nil {
			return err
		}
		engineDir = dir
	}
	if stateDir == "" {
		if !filepath.IsAbs(recordPath) {
			return errors.New("thenoise-provider: -records must be an absolute path")
		}
		stateDir = filepath.Join(filepath.Dir(recordPath), "thenoise-state")
	}
	leases := resourceclient.Discover()
	if resourceEndpoint != "" {
		leases = resourceclient.New(resourceEndpoint)
	}
	report := func(err error) { fmt.Fprintln(os.Stderr, "thenoise-provider:", err) }
	engineEndpoint := fmt.Sprintf("http://127.0.0.1:%d", port)
	managed, err := thenoise.NewRuntime(thenoise.ManagedConfig{
		Engine: thenoise.Config{Endpoint: engineEndpoint, Model: model, MaxRun: 10 * time.Minute},
		Launch: thenoise.LaunchConfig{RuntimeDir: engineDir, StateDir: stateDir, EngineZIP: engineZIP, DiT: dit, VAE: vae,
			TextEncoder: textEncoder, Device: device, Port: port,
			ReadyTimeout: 10 * time.Minute, ForceTimeout: 20 * time.Second},
		Leases: leases, Resource: resource, Amount: amount,
		Idle: 2 * time.Minute, Renew: 10 * time.Second, Wait: time.Second,
		OnError: report,
	})
	if err != nil {
		return err
	}
	host, err := thenoise.NewManagedHost(thenoise.HostConfig{Endpoint: endpoint,
		RuntimePrograms: []string{runtimeProgram}, StoreRoot: storeRoot,
		RecordPath: recordPath, AuditPath: auditPath, HostName: hostName, OnError: report}, managed)
	if err != nil {
		managed.Close()
		return err
	}
	go managed.Run(ctx)
	serveErr := host.Serve(ctx)
	managed.Close()
	if ctx.Err() != nil {
		return nil
	}
	return serveErr
}

// runEngine performs explicit local maintenance without constructing a host or
// starting the engine. Setup supplies its journal-owned paths to these commands.
func runEngine(ctx context.Context, args []string,
	install func(context.Context, string, string, string) error,
	unlock func(context.Context, string, string) error) error {
	if len(args) == 0 || (args[0] != "install" && args[0] != "unlock") {
		return errors.New("thenoise-provider: engine command is install or unlock")
	}
	flags := flag.NewFlagSet("thenoise-provider engine "+args[0], flag.ContinueOnError)
	engineDir := flags.String("engine-dir", "", "absolute pinned engine folder")
	stateDir := flags.String("state", "", "absolute private provider state folder")
	engineZIP := flags.String("engine-zip", "", "absolute pinned TheNoise ZIP (install only)")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("thenoise-provider: unexpected engine positional arguments")
	}
	if !filepath.IsAbs(*engineDir) || !filepath.IsAbs(*stateDir) {
		return errors.New("thenoise-provider: engine -engine-dir and -state must be absolute paths")
	}
	manifest := filepath.Join(*stateDir, thenoise.EngineFolderName+".manifest")
	if args[0] == "install" {
		if !filepath.IsAbs(*engineZIP) {
			return errors.New("thenoise-provider: engine install -engine-zip must be an absolute path")
		}
		return install(ctx, *engineZIP, *engineDir, manifest)
	}
	if *engineZIP != "" {
		return errors.New("thenoise-provider: -engine-zip belongs to engine install")
	}
	return unlock(ctx, *engineDir, manifest)
}
