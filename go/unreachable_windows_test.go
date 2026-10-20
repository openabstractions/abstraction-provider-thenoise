package thenoise

import (
	"context"
	"errors"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginehttp"
)

// An engine address every connect fails to reach, with an error a starting
// engine never gives, ends readiness after enginehttp.UnreachableFor, not
// after its whole budget. Port 0 stands in: a Windows connect to it reads
// WSAEADDRNOTAVAIL every time.
func TestReadyEndsOnAnUnreachableEndpoint(t *testing.T) {
	const endpoint = "http://127.0.0.1:0"
	client, err := enginehttp.NewClientWithVerifier(endpoint, func(identity.Process) (bool, error) { return true, nil })
	if err != nil {
		t.Fatal(err)
	}
	owner := &Runtime{cfg: ManagedConfig{Engine: Config{Endpoint: endpoint, Model: "flux_klein"},
		Launch: LaunchConfig{ReadyTimeout: time.Minute}}}
	process := &engineProcess{client: client, owner: owner}
	process.alive.Store(true)
	began := time.Now()
	err = process.Ready(context.Background())
	if !errors.Is(err, enginehttp.ErrEngineUnreachable) {
		t.Fatalf("got %v, want enginehttp.ErrEngineUnreachable", err)
	}
	if took := time.Since(began); took > enginehttp.UnreachableFor+5*time.Second {
		t.Fatalf("readiness ended after %s, want about %s", took, enginehttp.UnreachableFor)
	}
}
