package thenoise

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

func pngBytes(t *testing.T, size int) []byte {
	t.Helper()
	var buf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	img.Set(0, 0, color.RGBA{R: 1, A: 255})
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func newTestEngine(t *testing.T, endpoint string, stop func() error) *Engine {
	t.Helper()
	return newEngine(Config{Endpoint: endpoint, Model: "flux_klein", MaxRun: time.Minute}, &http.Client{}, stop)
}

func requestFor(mode wire.ImageMode) wire.ImageRequest {
	return wire.ImageRequest{Model: "flux_klein", Mode: mode, Prompt: "fox", Size: "512x512", Count: 1, Extensions: map[string]string{}}
}

func TestUnsupportedPreflightsBeforeCallingTheEngine(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Write(pngBytes(t, 512))
	}))
	defer server.Close()
	e := newTestEngine(t, server.URL, func() error { return nil })
	for _, req := range []wire.ImageRequest{
		requestFor(wire.ImageModeEdit),
		func() wire.ImageRequest { r := requestFor(wire.ImageModeGenerate); r.Count = 2; return r }(),
		func() wire.ImageRequest {
			r := requestFor(wire.ImageModeGenerate)
			r.Extensions["other/steps"] = "8"
			return r
		}(),
		func() wire.ImageRequest {
			r := requestFor(wire.ImageModeGenerate)
			r.RequiredExtensions = []string{"thenoise/steps"}
			return r
		}(),
	} {
		if e.Unsupported(req) == "" {
			t.Fatalf("request unexpectedly supported: %#v", req)
		}
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("unsupported preflight sent %d engine requests", got)
	}
}

func TestSupportedExtensionsMatchEngineValidationInventory(t *testing.T) {
	want := []string{
		"thenoise/negative-prompt",
		"thenoise/steps",
		"thenoise/guidance-scale",
		"thenoise/seed",
	}
	got := SupportedExtensions()
	if len(got) != len(want) {
		t.Fatalf("SupportedExtensions() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] || !supportedExtension(got[i]) {
			t.Fatalf("inventory item %d = %q, validation accepted=%v; want %q and accepted", i, got[i], supportedExtension(got[i]), want[i])
		}
	}
	got[0] = "other/injected"
	if !supportedExtension(want[0]) || supportedExtension("other/injected") {
		t.Fatal("mutating returned inventory changed engine validation")
	}
}

func TestRunMapsControlsAndReturnsBoundedPNG(t *testing.T) {
	want := pngBytes(t, 512)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/text2image" {
			t.Errorf("unexpected route %s %s", r.Method, r.URL.Path)
		}
		var got request
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Error(err)
		}
		if got.Prompt != "fox" || got.NegativePrompt != "blur" || got.Width != 512 || got.Height != 512 || got.Steps != 20 || got.GuidanceScale != 3.5 || got.Seed == nil || *got.Seed != 7 {
			t.Errorf("request mapping mismatch: %+v", got)
		}
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(want)
	}))
	defer server.Close()
	e := newTestEngine(t, server.URL, func() error { return nil })
	req := requestFor(wire.ImageModeGenerate)
	req.Extensions = map[string]string{"thenoise/negative-prompt": "blur", "thenoise/steps": "20", "thenoise/guidance-scale": "3.5", "thenoise/seed": "7"}
	var progress []string
	got := e.Run(context.Background(), "flux_klein", req, nil, func(_ int64, message string) { progress = append(progress, message) })
	if got.Outcome != wire.ReplyOutcomeCompleted || len(got.Images) != 1 || got.Images[0].MediaType != "image/png" || !bytes.Equal(got.Images[0].Data, want) {
		t.Fatalf("unexpected result: %+v", got)
	}
	if len(progress) != 3 || progress[0] != "waiting for engine" || progress[1] != "loading model" || progress[2] != "generating" {
		t.Fatalf("startup progress = %v", progress)
	}
}

func TestCancellationAfterAcquireDoesNotDispatchRender(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
	}))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	e := managedEngine(Config{Endpoint: server.URL, Model: "flux_klein", MaxRun: time.Minute},
		func(context.Context) (func(), *http.Client, func() error, error) {
			cancel()
			return func() {}, server.Client(), func() error { return nil }, nil
		})
	got := e.Run(ctx, "flux_klein", requestFor(wire.ImageModeGenerate), nil, nil)
	if got.Outcome != wire.ReplyOutcomeCancelled || requests.Load() != 0 {
		t.Fatalf("cancelled acquisition dispatched render: outcome=%v requests=%d", got.Outcome, requests.Load())
	}
}

func TestRunRejectsCorruptPNGBody(t *testing.T) {
	bad := pngBytes(t, 512)
	idat := bytes.Index(bad, []byte("IDAT"))
	if idat < 0 || idat+5 >= len(bad) {
		t.Fatal("fixture has no IDAT payload")
	}
	bad[idat+5] ^= 0xff
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(bad)
	}))
	defer server.Close()
	e := newTestEngine(t, server.URL, func() error { return nil })
	got := e.Run(context.Background(), "flux_klein", requestFor(wire.ImageModeGenerate), nil, nil)
	if got.Outcome != wire.ReplyOutcomeUnavailable || len(got.Images) != 0 {
		t.Fatalf("corrupt PNG was published: %+v", got)
	}
}

func TestCancellationWaitsForConfirmedEngineExit(t *testing.T) {
	started, disconnect, backendStopped := make(chan struct{}), make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		_ = conn.Close()
		close(disconnect)
		<-backendStopped
	}))
	defer server.Close()
	exited := make(chan struct{})
	e := newTestEngine(t, server.URL, func() error { <-exited; close(backendStopped); return nil })
	ctx, cancel := context.WithCancel(context.Background())
	finished := make(chan struct{})
	go func() {
		e.Run(ctx, "flux_klein", requestFor(wire.ImageModeGenerate), nil, nil)
		close(finished)
	}()
	<-started
	<-disconnect
	cancel()
	select {
	case <-finished:
		t.Fatal("Run returned before stop-and-wait confirmed engine exit")
	case <-time.After(50 * time.Millisecond):
	}
	close(exited)
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not return after confirmed engine exit")
	}
	<-backendStopped
}

func TestConcurrentRunsAreSerializedAndWaitingRunCanCancel(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var requests, stops atomic.Int32
	img := pngBytes(t, 512)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requests.Add(1) == 1 {
			close(started)
		}
		<-release
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(img)
	}))
	defer server.Close()
	e := newTestEngine(t, server.URL, func() error { stops.Add(1); return nil })
	firstDone := make(chan inference.ImageEngineReply, 1)
	go func() {
		firstDone <- e.Run(context.Background(), "flux_klein", requestFor(wire.ImageModeGenerate), nil, nil)
	}()
	<-started

	ctx, cancel := context.WithCancel(context.Background())
	secondDone := make(chan inference.ImageEngineReply, 1)
	go func() { secondDone <- e.Run(ctx, "flux_klein", requestFor(wire.ImageModeGenerate), nil, nil) }()
	cancel()
	select {
	case got := <-secondDone:
		if got.Outcome != wire.ReplyOutcomeCancelled {
			t.Fatalf("waiting run outcome = %v, want cancelled", got.Outcome)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled waiting run did not return")
	}
	if got := requests.Load(); got != 1 {
		t.Fatalf("waiting run reached engine: requests=%d, want 1", got)
	}
	if got := stops.Load(); got != 0 {
		t.Fatalf("cancelling waiting run stopped shared engine %d times", got)
	}
	close(release)
	select {
	case got := <-firstDone:
		if got.Outcome != wire.ReplyOutcomeCompleted {
			t.Fatalf("first run outcome = %v, want completed", got.Outcome)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("first run did not finish")
	}
}
