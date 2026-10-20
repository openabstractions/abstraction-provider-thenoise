//go:build !windows

package thenoise

import "errors"

func kernelWindowsDir() (string, error) { return "", errors.New("no Windows folder outside Windows") }
