package thenoise

import (
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"testing"
)

// hostileParent is a provider environment carrying credentials, proxies,
// interpreter hooks, the variables condition 11 forbids and planted system
// and profile locations. The engine's environment is built from the system
// and receives none of it.
var hostileParent = []string{
	`SystemRoot=C:\planted`, `USERPROFILE=C:\planted\u`, `LOCALAPPDATA=C:\planted\Local`, `TEMP=C:\planted\Temp`,
	`ComSpec=C:\planted\cmd.exe`, `PATHEXT=.PY`, `ProgramFiles=C:\planted`, `COMPUTERNAME=planted`,
	`Path=C:\evil;C:\Windows\System32`, "HF_TOKEN=hf_secret", "HUGGING_FACE_HUB_TOKEN=hf_secret", "GITHUB_TOKEN=ghp_secret",
	"OPENAI_API_KEY=sk-secret", "AWS_SECRET_ACCESS_KEY=secret", "HTTPS_PROXY=http://proxy:8080", `PYTHONPATH=C:\evil`,
	`PYTHONHOME=C:\evil`, "PYTHONDONTWRITEBYTECODE=0", "HF_HUB_OFFLINE=0", "TORCH_BLAS_PREFER_HIPBLASLT=0", "ROCBLAS_USE_HIPBLASLT=0",
	"PYTORCH_TUNABLEOP_ENABLED=1", "TORCH_COMPILE_DISABLE=0", `HIPBLASLT_TENSILE_LIBPATH=C:\evil\gfx1151`,
	`ROCBLAS_TENSILE_LIBPATH=C:\evil\gfx1151`, `AMD_COMGR_CACHE_DIR=C:\shared\comgr`, "MIOPEN_DISABLE_CACHE=0",
}

func plantParent(t *testing.T) {
	t.Helper()
	for _, kv := range hostileParent {
		name, value, _ := strings.Cut(kv, "=")
		t.Setenv(name, value)
	}
}

func envMap(env []string) map[string]string {
	m := map[string]string{}
	for _, kv := range env {
		name, value, _ := strings.Cut(kv, "=")
		m[name] = value
	}
	return m
}

func sortedNames(env map[string]string) []string {
	names := make([]string, 0, len(env))
	for name := range env {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// The engine environment is the provider kit's clean environment with the
// reviewed variables and the private caches. Every system variable comes
// from the kernel, the token or the registry; a planted value in the
// provider's environment never reaches the engine. Failures print names.
func TestCPUEnvironmentIsTheReviewedOne(t *testing.T) {
	state := t.TempDir()
	plantParent(t)
	var omitted []error
	env, err := launchEnv(state, nil, func(err error) { omitted = append(omitted, err) })
	if err != nil {
		t.Fatal(err)
	}
	got := envMap(env)
	want := map[string]string{
		"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
		"PYTHONDONTWRITEBYTECODE": "1",
		"AMD_COMGR_CACHE_DIR":     filepath.Join(state, "rocm-cache", "comgr"),
		"MIOPEN_USER_DB_PATH":     filepath.Join(state, "rocm-cache", "miopen-db"),
		"MIOPEN_CUSTOM_CACHE_DIR": filepath.Join(state, "rocm-cache", "miopen-cache"),
	}
	for name, value := range want {
		if got[name] != value {
			t.Fatalf("%s is not the reviewed value (names %q)", name, sortedNames(got))
		}
	}
	for _, kv := range hostileParent {
		_, planted, _ := strings.Cut(kv, "=")
		for name, value := range got {
			if value == planted && want[name] != value {
				t.Fatalf("%s carries a value from the provider's environment", name)
			}
		}
	}
	temp := engineTemp(state)
	if runtime.GOOS != "windows" {
		names := append([]string{"PATH", "TMPDIR"}, sortedNames(want)...)
		sort.Strings(names)
		if !slices.Equal(sortedNames(got), names) || got["TMPDIR"] != temp || got["PATH"] != "/usr/bin:/bin" {
			t.Fatalf("environment names %q", sortedNames(got))
		}
		return
	}
	windowsDir, err := kernelWindowsDir()
	if err != nil {
		t.Fatal(err)
	}
	if got["SystemRoot"] != windowsDir || got["windir"] != windowsDir || got["TEMP"] != temp || got["TMP"] != temp {
		t.Fatal("SystemRoot, windir, TEMP or TMP is not the system's or the provider's private folder")
	}
	if got["PATH"] != filepath.Join(windowsDir, "System32")+";"+windowsDir+";"+filepath.Join(windowsDir, "System32", "Wbem") {
		t.Fatal("PATH is not the Windows system folders")
	}
	expected := append([]string{"PATH", "SystemDrive", "SystemRoot", "TEMP", "TMP", "windir"}, systemVariables...)
	expected = append(expected, sortedNames(want)...)
	if len(got)+len(omitted) != len(expected) {
		t.Fatalf("environment names %q with %d left out, wanted %d names", sortedNames(got), len(omitted), len(expected))
	}
	for _, name := range expected {
		if _, ok := got[name]; !ok && len(omitted) == 0 {
			t.Fatalf("%s is missing and was not reported", name)
		}
	}
}

func TestGPUEnvironmentAddsOnlyTheGateVariables(t *testing.T) {
	state := t.TempDir()
	plantParent(t)
	gpu := []string{`HIPBLASLT_TENSILE_LIBPATH=C:\E\LIB~1\GFX115~1`, `ROCBLAS_TENSILE_LIBPATH=C:\E\LIB~2\GFX115~1`, "ROCBLAS_USE_HIPBLASLT=0"}
	cpuEnv, err := launchEnv(state, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	gpuEnv, err := launchEnv(state, gpu, nil)
	if err != nil {
		t.Fatal(err)
	}
	cpu, got := envMap(cpuEnv), envMap(gpuEnv)
	if got["HIPBLASLT_TENSILE_LIBPATH"] != `C:\E\LIB~1\GFX115~1` || got["ROCBLAS_TENSILE_LIBPATH"] != `C:\E\LIB~2\GFX115~1` {
		t.Fatalf("gate variables missing: %q", sortedNames(got))
	}
	if len(got) != len(cpu)+2 {
		t.Fatalf("GPU environment %q adds more than the two gate variables to %q", sortedNames(got), sortedNames(cpu))
	}
	for _, forbidden := range []string{"ROCBLAS_USE_HIPBLASLT", "PYTORCH_TUNABLEOP_ENABLED", "TORCH_BLAS_PREFER_HIPBLASLT", "TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE"} {
		if _, ok := got[forbidden]; ok {
			t.Fatalf("%s reached the engine environment", forbidden)
		}
	}
}
