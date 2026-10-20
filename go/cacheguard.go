package thenoise

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

// cacheFolders are the ROCm caches the GPU path writes, moved out of the user
// profile (%LOCALAPPDATA%\comgr, %USERPROFILE%\.miopen) into the provider's
// state folder (security review condition 10). They hold GPU code objects the
// engine loads without a content check. Sources: AMD_COMGR_CACHE_DIR in
// comgr's README (ROCm/llvm-project amd/comgr/README.md); MIOPEN_USER_DB_PATH
// in projects/miopen/src/db_path.cpp.in and MIOPEN_CUSTOM_CACHE_DIR in
// projects/miopen/src/binary_cache.cpp of ROCm/rocm-libraries. All three names
// occur in the bundled amd_comgr.dll and MIOpen.dll.
var cacheFolders = [...]struct{ variable, folder string }{
	{"AMD_COMGR_CACHE_DIR", "comgr"},
	{"MIOPEN_USER_DB_PATH", "miopen-db"},
	{"MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache"},
}

// cacheRoot is the folder below the provider state that holds the ROCm caches.
func cacheRoot(stateDir string) string { return filepath.Join(stateDir, "rocm-cache") }

// cacheStamp records which bundle and account filled the caches. It sits in
// the state folder beside the cache root, outside the cache folders the
// engine writes.
func cacheStamp(stateDir string) string { return filepath.Join(stateDir, "rocm-cache.stamp") }

// cacheSecurity holds the account and ACL operations of the cache guard.
// Production uses the Windows implementations; tests substitute them.
type cacheSecurity struct {
	// account is the SID of the provider's account.
	account func() (string, error)
	// restrict gives a folder an explicit, protected DACL granting full
	// access to the account, SYSTEM and Administrators only.
	restrict func(folder, sid string) error
	// private reports whether a folder's DACL is exactly that ACL.
	private func(folder, sid string) (bool, error)
	// owner is the owner SID of a file or folder.
	owner func(path string) (string, error)
}

// prepareCaches keeps the provider's ROCm cache folders private. It empties
// them when the bundle or the account differs from the stamp, when a folder's
// ACL is not the private one, or when any entry has another owner or is a
// link, and then recreates them with the private ACL. It returns the reason
// the caches were emptied, or "" when they were kept.
func prepareCaches(stateDir, bundle string, sec cacheSecurity) (string, error) {
	sid, err := sec.account()
	if err != nil {
		return "", fmt.Errorf("thenoise: reading the provider account: %w", err)
	}
	stamp := "bundle=" + bundle + "\naccount=" + sid + "\n"
	reason := cacheDrift(stateDir, stamp, sid, sec)
	if reason == "" {
		return "", nil
	}
	root := cacheRoot(stateDir)
	if err := os.RemoveAll(root); err != nil {
		return reason, fmt.Errorf("thenoise: emptying ROCm caches: %w", err)
	}
	if err := os.Remove(cacheStamp(stateDir)); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return reason, err
	}
	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		return reason, err
	}
	if err := os.Mkdir(root, 0o700); err != nil {
		return reason, err
	}
	if err := sec.restrict(root, sid); err != nil {
		return reason, fmt.Errorf("thenoise: restricting ROCm cache folder: %w", err)
	}
	for _, c := range cacheFolders {
		folder := filepath.Join(root, c.folder)
		if err := os.Mkdir(folder, 0o700); err != nil {
			return reason, err
		}
		if err := sec.restrict(folder, sid); err != nil {
			return reason, fmt.Errorf("thenoise: restricting ROCm cache folder: %w", err)
		}
	}
	if err := writeFileAtomic(cacheStamp(stateDir), []byte(stamp)); err != nil {
		return reason, err
	}
	return reason, nil
}

// cachesNew is the reason for the first creation of the caches.
const cachesNew = "no cache stamp was recorded"

// cacheDrift names the first reason the caches cannot be trusted, or "".
func cacheDrift(stateDir, stamp, sid string, sec cacheSecurity) string {
	recorded, err := os.ReadFile(cacheStamp(stateDir))
	if err != nil {
		return cachesNew
	}
	if string(recorded) != stamp {
		return "the bundle or the account changed"
	}
	root := cacheRoot(stateDir)
	folders := []string{root}
	for _, c := range cacheFolders {
		folders = append(folders, filepath.Join(root, c.folder))
	}
	for _, folder := range folders {
		info, err := os.Lstat(folder)
		if err != nil || !info.IsDir() {
			return "a cache folder is missing"
		}
		ok, err := sec.private(folder, sid)
		if err != nil || !ok {
			return "a cache folder's ACL changed"
		}
	}
	reason := ""
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			reason = "a cache entry could not be read"
			return fs.SkipAll
		}
		if t := d.Type(); t&fs.ModeType != 0 && t&fs.ModeType != fs.ModeDir {
			reason = "a cache entry is a link"
			return fs.SkipAll
		}
		owner, err := sec.owner(p)
		if err != nil || owner != sid {
			reason = "a cache file has another owner"
			return fs.SkipAll
		}
		return nil
	})
	return reason
}
