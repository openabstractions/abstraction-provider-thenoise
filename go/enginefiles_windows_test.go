package thenoise

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"golang.org/x/sys/windows"
)

// An engine file another program holds open without sharing refuses the
// launch as unreadable, with the sharing violation and without the reinstall
// step a changed tree names; the next launch passes once the file is free.
func TestALockedEngineFileRefusesOnlyThatLaunch(t *testing.T) {
	engineDir, stateDir, pin := installSmall(t)
	v := testFiles()
	name, err := windows.UTF16PtrFromString(filepath.Join(engineDir, "thenoise.bat"))
	if err != nil {
		t.Fatal(err)
	}
	lock, err := windows.CreateFile(name, windows.GENERIC_READ, 0, nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = launchEngine(t.Context(), v, engineDir, manifestFile(stateDir), pin)
	windows.CloseHandle(lock)
	if !errors.Is(err, enginefiles.ErrUnreadable) || !errors.Is(err, windows.ERROR_SHARING_VIOLATION) || errors.Is(err, enginefiles.ErrMismatch) ||
		strings.Contains(err.Error(), "-engine-zip") || !strings.Contains(err.Error(), "thenoise.bat") {
		t.Fatalf("a locked engine file: %v", err)
	}
	h, err := launchEngine(t.Context(), v, engineDir, manifestFile(stateDir), pin)
	if err != nil {
		t.Fatalf("the freed engine: %v", err)
	}
	h.Release()
}
