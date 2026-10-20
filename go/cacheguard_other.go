//go:build !windows

package thenoise

import (
	"errors"
	"os"
	"strconv"
	"syscall"
)

// Outside Windows the cache folders are private through their 0700 mode and
// the owner is the numeric user ID.
func systemCacheSecurity() cacheSecurity {
	return cacheSecurity{
		account:  func() (string, error) { return strconv.Itoa(os.Getuid()), nil },
		restrict: func(folder, _ string) error { return os.Chmod(folder, 0o700) },
		private: func(folder, _ string) (bool, error) {
			info, err := os.Stat(folder)
			if err != nil {
				return false, err
			}
			return info.Mode().Perm() == 0o700, nil
		},
		owner: func(path string) (string, error) {
			info, err := os.Lstat(path)
			if err != nil {
				return "", err
			}
			st, ok := info.Sys().(*syscall.Stat_t)
			if !ok {
				return "", errors.New("thenoise: no owner information")
			}
			return strconv.FormatUint(uint64(st.Uid), 10), nil
		},
	}
}
