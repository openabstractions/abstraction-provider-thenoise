package thenoise

import (
	"os"
	"path/filepath"
	"testing"
)

// fakeSecurity is an account "S-1-5-21-1" whose folders stay private and
// whose files it owns, until a test changes the answers.
type fakeSecurity struct {
	sid        string
	restricted map[string]int
	public     map[string]bool
	foreign    map[string]bool
}

func newFakeSecurity() *fakeSecurity {
	return &fakeSecurity{sid: "S-1-5-21-1", restricted: map[string]int{}, public: map[string]bool{}, foreign: map[string]bool{}}
}

func (f *fakeSecurity) sec() cacheSecurity {
	return cacheSecurity{
		account:  func() (string, error) { return f.sid, nil },
		restrict: func(folder, _ string) error { f.restricted[folder]++; delete(f.public, folder); return nil },
		private:  func(folder, _ string) (bool, error) { return !f.public[folder], nil },
		owner: func(p string) (string, error) {
			if f.foreign[p] {
				return "S-1-5-21-2", nil
			}
			return f.sid, nil
		},
	}
}

// plant writes a cache entry and reports whether a later check finds it.
func plant(t *testing.T, stateDir string) func() bool {
	t.Helper()
	file := filepath.Join(cacheRoot(stateDir), "comgr", "llvmcache-1")
	if err := os.WriteFile(file, []byte("\x7fELF"), 0o600); err != nil {
		t.Fatal(err)
	}
	return func() bool { _, err := os.Stat(file); return err == nil }
}

func TestCachesAreCreatedPrivateAndKept(t *testing.T) {
	state := t.TempDir()
	f := newFakeSecurity()
	reason, err := prepareCaches(state, "bundle-a", f.sec())
	if err != nil || reason != cachesNew {
		t.Fatalf("first preparation: %q %v", reason, err)
	}
	for _, folder := range []string{cacheRoot(state), filepath.Join(cacheRoot(state), "comgr"), filepath.Join(cacheRoot(state), "miopen-db"), filepath.Join(cacheRoot(state), "miopen-cache")} {
		if f.restricted[folder] != 1 {
			t.Fatalf("%s restricted %d times", folder, f.restricted[folder])
		}
	}
	present := plant(t, state)
	if reason, err := prepareCaches(state, "bundle-a", f.sec()); err != nil || reason != "" || !present() {
		t.Fatalf("unchanged caches were emptied: %q %v", reason, err)
	}
}

func TestCachesAreEmptiedWhenTheBundleChanges(t *testing.T) {
	state := t.TempDir()
	f := newFakeSecurity()
	if _, err := prepareCaches(state, "bundle-a", f.sec()); err != nil {
		t.Fatal(err)
	}
	present := plant(t, state)
	if reason, err := prepareCaches(state, "bundle-b", f.sec()); err != nil || reason == "" || present() {
		t.Fatalf("caches survived a bundle change: %q %v", reason, err)
	}
}

func TestCachesAreEmptiedWhenTheAccountChanges(t *testing.T) {
	state := t.TempDir()
	f := newFakeSecurity()
	if _, err := prepareCaches(state, "bundle-a", f.sec()); err != nil {
		t.Fatal(err)
	}
	present := plant(t, state)
	f.sid = "S-1-5-21-9"
	if reason, err := prepareCaches(state, "bundle-a", f.sec()); err != nil || reason == "" || present() {
		t.Fatalf("caches survived an account change: %q %v", reason, err)
	}
}

func TestCachesAreEmptiedWhenAnACLChanges(t *testing.T) {
	state := t.TempDir()
	f := newFakeSecurity()
	if _, err := prepareCaches(state, "bundle-a", f.sec()); err != nil {
		t.Fatal(err)
	}
	present := plant(t, state)
	miopen := filepath.Join(cacheRoot(state), "miopen-cache")
	f.public[miopen] = true
	if reason, err := prepareCaches(state, "bundle-a", f.sec()); err != nil || reason == "" || present() {
		t.Fatalf("caches survived an ACL change: %q %v", reason, err)
	}
	if f.public[miopen] || f.restricted[miopen] != 2 {
		t.Fatal("the recreated folder did not get the private ACL again")
	}
}

func TestCachesAreEmptiedWhenAFileHasAnotherOwner(t *testing.T) {
	state := t.TempDir()
	f := newFakeSecurity()
	if _, err := prepareCaches(state, "bundle-a", f.sec()); err != nil {
		t.Fatal(err)
	}
	present := plant(t, state)
	f.foreign[filepath.Join(cacheRoot(state), "comgr", "llvmcache-1")] = true
	if reason, err := prepareCaches(state, "bundle-a", f.sec()); err != nil || reason == "" || present() {
		t.Fatalf("a foreign-owned cache file survived: %q %v", reason, err)
	}
}
