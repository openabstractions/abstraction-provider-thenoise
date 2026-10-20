//go:build windows

package thenoise

import "golang.org/x/sys/windows"

// kernelWindowsDir is the Windows folder the kernel names, which the engine
// environment must carry.
func kernelWindowsDir() (string, error) { return windows.GetSystemWindowsDirectory() }
