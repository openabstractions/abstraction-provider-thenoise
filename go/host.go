package thenoise

import (
	"errors"

	inference "github.com/openabstractions/abstraction-inference/go"
	"github.com/openabstractions/abstraction-provider-kit/go/providerhost"
	router "github.com/openabstractions/abstraction-router/go"
)

// HostConfig names one provider-local endpoint and store. The designated OA
// runtime mediates the resulting image and copies it to its caller-owned store.
type HostConfig struct {
	Endpoint        string
	RuntimePrograms []string
	StoreRoot       string
	RecordPath      string
	HostName        string
	// AuditPath is the provider's persistent audit journal; empty selects
	// providerhost.DefaultAuditPath(RecordPath).
	AuditPath string
	// OnError receives provider faults, including a failed audit append.
	OnError func(error)
}

// NewManagedHost composes the standard authenticated image/content leaf with
// one lazy TheNoise runtime. It declares only the exact configured model.
func NewManagedHost(cfg HostConfig, managed *Runtime) (*providerhost.Host, error) {
	if managed == nil || cfg.HostName == "" {
		return nil, errors.New("thenoise: managed runtime and host name required")
	}
	declaration, err := router.NewImageEngine(cfg.HostName, []string{managed.cfg.Engine.Model})
	if err != nil {
		return nil, err
	}
	host, err := providerhost.NewImage(providerhost.ImageConfig{
		Endpoint: cfg.Endpoint, RuntimePrograms: cfg.RuntimePrograms,
		Host: declaration, Engine: managed.Engine(), Admit: managed.Admit,
		StoreRoot: cfg.StoreRoot, RecordPath: cfg.RecordPath,
		AuditPath: cfg.AuditPath, OnError: cfg.OnError,
	})
	if err != nil {
		return nil, err
	}
	if err := managed.BindActivity(host.Provider.ActiveForModel); err != nil {
		_ = host.Close()
		return nil, err
	}
	return host, nil
}

var _ inference.ImageEngine = (*Engine)(nil)
