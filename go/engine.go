// Package thenoise adapts the TheNoise local HTTP image API to OA's native
// inference image-engine callback. The executable and model remain operator
// supplied; process ownership and identity-bound HTTP are injected by the host.
package thenoise

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/openabstractions/abstraction-facade/go/supervision"
	identity "github.com/openabstractions/abstraction-identity"
	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-provider-kit/go/enginehttp"
)

const maxImageBytes = 20 << 20
const maxReplyBytes = maxImageBytes + 1

// imageExtensions is the engine-owned inventory used both for request
// validation and provider declaration metadata. Keep this list limited to
// controls translated by extensionValue below.
var imageExtensions = [...]string{
	"thenoise/negative-prompt",
	"thenoise/steps",
	"thenoise/guidance-scale",
	"thenoise/seed",
}

// SupportedExtensions returns the image extension keys implemented by this
// engine. The returned slice is a copy so callers cannot mutate the inventory.
func SupportedExtensions() []string {
	return append([]string(nil), imageExtensions[:]...)
}

// Config binds one running TheNoise instance to one declared model and one
// process generation. HTTP must be the identity-bound client built by the
// platform supervisor; StopAndWait must confirm the entire owned process tree
// has exited before an uncertain/cancelled request is released.
type Config struct {
	Endpoint string
	Model    string
	Process  *supervision.Process
	MaxRun   time.Duration
}

// Engine is a serialized adapter for the currently loaded TheNoise model.
type Engine struct {
	cfg          Config
	acquire      func(context.Context) (func(), *http.Client, func() error, error)
	poisonOnStop bool
	stopped      atomic.Bool
	slot         chan struct{}
}

func New(cfg Config) (*Engine, error) {
	if cfg.Process == nil || cfg.Model == "" || cfg.MaxRun <= 0 || cfg.Process.PID() <= 0 || cfg.Process.StartTime().IsZero() {
		return nil, errors.New("thenoise: require model, owned process generation and max run")
	}
	u, err := url.Parse(cfg.Endpoint)
	if err != nil || u.Scheme != "http" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Path != "" || u.Hostname() != "127.0.0.1" || u.Port() == "" {
		return nil, errors.New("thenoise: endpoint must be a 127.0.0.1 HTTP origin with an explicit port")
	}
	client, err := enginehttp.NewClientWithVerifier(cfg.Endpoint, func(peer identity.Process) (bool, error) {
		return cfg.Process.Contains(peer.PID, peer.StartTime)
	})
	if err != nil {
		return nil, err
	}
	cfg.Endpoint = strings.TrimRight(cfg.Endpoint, "/")
	stop := func() error {
		_ = cfg.Process.Stop()
		_, waitErr := cfg.Process.Wait()
		return waitErr
	}
	return newEngine(cfg, client, stop), nil
}

func newEngine(cfg Config, client *http.Client, stop func() error) *Engine {
	cfg.Process = nil
	return &Engine{cfg: cfg, acquire: func(context.Context) (func(), *http.Client, func() error, error) {
		return func() {}, client, stop, nil
	}, poisonOnStop: true, slot: make(chan struct{}, 1)}
}

func managedEngine(cfg Config, acquire func(context.Context) (func(), *http.Client, func() error, error)) *Engine {
	cfg.Process = nil
	return &Engine{cfg: cfg, acquire: acquire, slot: make(chan struct{}, 1)}
}

type request struct {
	Prompt         string  `json:"prompt"`
	NegativePrompt string  `json:"negative_prompt,omitempty"`
	Width          int     `json:"width,omitempty"`
	Height         int     `json:"height,omitempty"`
	Steps          int     `json:"steps,omitempty"`
	GuidanceScale  float64 `json:"guidance_scale,omitempty"`
	Seed           *int64  `json:"seed,omitempty"`
}

type result struct {
	reply inference.ImageEngineReply
	stop  bool
}

// Unsupported performs request-only checks; it never touches TheNoise.
func (e *Engine) Unsupported(req wire.ImageRequest) string {
	if e.stopped.Load() {
		return "engine:stopped"
	}
	if req.Model != e.cfg.Model {
		return "model:" + req.Model
	}
	if req.Mode != wire.ImageModeGenerate {
		return "mode:edit"
	}
	if req.Count != 1 {
		return "count"
	}
	if req.Prompt == "" || len(req.Prompt) > 32768 {
		return "prompt"
	}
	if req.Size != "auto" {
		w, h, ok := strings.Cut(req.Size, "x")
		if !ok {
			return "size"
		}
		width, errW := strconv.Atoi(w)
		height, errH := strconv.Atoi(h)
		if errW != nil || errH != nil || width < 64 || width > 4096 || height < 64 || height > 4096 || width%8 != 0 || height%8 != 0 {
			return "size"
		}
	}
	for _, key := range req.RequiredExtensions {
		if !supportedExtension(key) {
			return "extension:" + key
		}
		if _, present := req.Extensions[key]; !present {
			return "extension:" + key
		}
	}
	for key, value := range req.Extensions {
		if !supportedExtension(key) || len(value) > 32768 {
			return "extension:" + key
		}
		if _, err := extensionValue(key, value); err != nil {
			return "extension:" + key
		}
	}
	return ""
}

func supportedExtension(key string) bool {
	for _, supported := range imageExtensions {
		if key == supported {
			return true
		}
	}
	return false
}

func extensionValue(key, value string) (any, error) {
	switch key {
	case "thenoise/negative-prompt":
		return value, nil
	case "thenoise/steps":
		n, err := strconv.Atoi(value)
		if err != nil || n < 1 || n > 100 {
			return nil, errors.New("steps out of range")
		}
		return n, nil
	case "thenoise/guidance-scale":
		f, err := strconv.ParseFloat(value, 64)
		if err != nil || f < 0 || f > 100 {
			return nil, errors.New("guidance out of range")
		}
		return f, nil
	case "thenoise/seed":
		return strconv.ParseInt(value, 10, 64)
	default:
		return nil, errors.New("unknown extension")
	}
}

// Run holds the operation until the request ends or StopAndWait proves the
// engine tree exited. An HTTP cancellation by itself is not compute completion.
func (e *Engine) Run(ctx context.Context, model string, req wire.ImageRequest, _ map[string]inference.ImageInput, progress func(int64, string)) inference.ImageEngineReply {
	emit := func(percent int64, message string) {
		if progress != nil {
			progress(percent, message)
		}
	}
	if reason := e.Unsupported(req); reason != "" || model != e.cfg.Model {
		if reason == "" {
			reason = "model:" + model
		}
		return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeUnsupportedFeature, Reason: reason}
	}
	if err := ctx.Err(); err != nil {
		return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeCancelled, Reason: "caller"}
	}
	emit(0, "waiting for engine")
	select {
	case e.slot <- struct{}{}:
		defer func() { <-e.slot }()
	case <-ctx.Done():
		return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeCancelled, Reason: "caller"}
	}
	emit(1, "loading model")
	end, client, stop, err := e.acquire(ctx)
	if err != nil {
		if ctx.Err() != nil {
			return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeCancelled, Reason: "caller"}
		}
		if errors.Is(err, ErrGPUPathTooDeep) {
			return unavailable("engine:gpu-path-too-deep")
		}
		if errors.Is(err, ErrGPUCrashed) {
			return unavailable("engine:gpu-crashed")
		}
		return unavailable("engine:unavailable")
	}
	if end != nil {
		defer end()
	}
	if client == nil || stop == nil {
		return unavailable("engine:unavailable")
	}
	if ctx.Err() != nil {
		return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeCancelled, Reason: "caller"}
	}
	emit(5, "generating")
	body := request{Prompt: req.Prompt}
	if req.Size != "auto" {
		w, h, _ := strings.Cut(req.Size, "x")
		body.Width, _ = strconv.Atoi(w)
		body.Height, _ = strconv.Atoi(h)
	}
	for key, value := range req.Extensions {
		switch key {
		case "thenoise/negative-prompt":
			body.NegativePrompt = value
		case "thenoise/steps":
			body.Steps, _ = strconv.Atoi(value)
		case "thenoise/guidance-scale":
			body.GuidanceScale, _ = strconv.ParseFloat(value, 64)
		case "thenoise/seed":
			seed, _ := strconv.ParseInt(value, 10, 64)
			body.Seed = &seed
		}
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return unavailable("request:encode")
	}
	requestCtx, cancel := context.WithTimeout(context.Background(), e.cfg.MaxRun)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(requestCtx, http.MethodPost, e.cfg.Endpoint+"/text2image", bytes.NewReader(payload))
	if err != nil {
		return unavailable("request:create")
	}
	httpReq.Header.Set("Content-Type", "application/json")
	finished := make(chan result, 1)
	go func() { finished <- e.do(client, httpReq, body.Width, body.Height) }()
	var tick *time.Ticker
	var retry <-chan time.Time
	callerDone, requestDone := ctx.Done(), requestCtx.Done()
	for {
		select {
		case got := <-finished:
			if !got.stop {
				if tick != nil {
					tick.Stop()
				}
				return got.reply
			}
			callerDone, requestDone = nil, nil
			if err := stop(); err == nil {
				if e.poisonOnStop {
					e.stopped.Store(true)
				}
				cancel()
				if tick != nil {
					tick.Stop()
				}
				if ctx.Err() != nil {
					return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeCancelled, Reason: "caller"}
				}
				return unavailable("engine:request-ended")
			}
			if tick == nil {
				tick = time.NewTicker(time.Second)
				retry = tick.C
			}
		case <-callerDone:
			callerDone, requestDone = nil, nil
			if err := stop(); err == nil {
				if e.poisonOnStop {
					e.stopped.Store(true)
				}
				cancel()
				<-finished
				if tick != nil {
					tick.Stop()
				}
				return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeCancelled, Reason: "caller"}
			}
			if tick == nil {
				tick = time.NewTicker(time.Second)
				retry = tick.C
			}
		case <-requestDone:
			callerDone, requestDone = nil, nil
			if err := stop(); err == nil {
				if e.poisonOnStop {
					e.stopped.Store(true)
				}
				cancel()
				<-finished
				if tick != nil {
					tick.Stop()
				}
				return unavailable("engine:timeout")
			}
			if tick == nil {
				tick = time.NewTicker(time.Second)
				retry = tick.C
			}
		case <-retry:
			if err := stop(); err == nil {
				if e.poisonOnStop {
					e.stopped.Store(true)
				}
				cancel()
				<-finished
				tick.Stop()
				if ctx.Err() != nil {
					return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeCancelled, Reason: "caller"}
				}
				return unavailable("engine:timeout")
			}
		}
	}
}

func (e *Engine) do(client *http.Client, req *http.Request, width, height int) result {
	resp, err := client.Do(req)
	if err != nil {
		return result{stop: true}
	}
	defer resp.Body.Close()
	mediaType, _, mediaErr := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if resp.StatusCode != http.StatusOK || mediaErr != nil || mediaType != "image/png" {
		return result{reply: unavailable(fmt.Sprintf("engine:http:%d", resp.StatusCode))}
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxReplyBytes))
	if err != nil || len(data) == 0 || len(data) > maxImageBytes || http.DetectContentType(data) != "image/png" {
		return result{reply: unavailable("engine:image")}
	}
	config, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil || config.Width < 64 || config.Height < 64 || config.Width > 4096 || config.Height > 4096 || config.Width%8 != 0 || config.Height%8 != 0 || width > 0 && config.Width != width || height > 0 && config.Height != height {
		return result{reply: unavailable("engine:image")}
	}
	// DecodeConfig reads only headers. Decode the bounded image as well so a
	// truncated or corrupt IDAT cannot be committed as a valid PNG delivery.
	if _, err := png.Decode(bytes.NewReader(data)); err != nil {
		return result{reply: unavailable("engine:image")}
	}
	return result{reply: inference.ImageEngineReply{Outcome: wire.ReplyOutcomeCompleted, Images: []inference.ImageOutput{{Data: data, MediaType: "image/png"}}}}
}

func unavailable(reason string) inference.ImageEngineReply {
	return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeUnavailable, Reason: reason}
}
