//go:build windows

package thenoise

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"golang.org/x/sys/windows"
)

// privateSDDL is the ACL restrictFolder writes: full control, inherited by
// files and folders, for the account, SYSTEM and Administrators.
func privateSDDL(sid string) string {
	return "D:P(A;OICI;FA;;;" + sid + ")(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
}

func TestRealPrivateACLIsWrittenAndChecked(t *testing.T) {
	sid, err := enginefiles.AccountSID()
	if err != nil {
		t.Fatal(err)
	}
	folder := filepath.Join(t.TempDir(), "cache")
	if err := os.Mkdir(folder, 0o700); err != nil {
		t.Fatal(err)
	}
	if ok, err := folderPrivate(folder, sid); err != nil || ok {
		t.Fatalf("an inherited temp-folder ACL counted as private: %v %v", ok, err)
	}
	if err := restrictFolder(folder, sid); err != nil {
		t.Fatal(err)
	}
	if ok, err := folderPrivate(folder, sid); err != nil || !ok {
		t.Fatalf("the private ACL was not recognized: %v %v", ok, err)
	}
	if ok, _ := folderPrivate(folder, "S-1-5-21-1-2-3-4"); ok {
		t.Fatal("the private ACL of another account was accepted")
	}
	file := filepath.Join(folder, "llvmcache-1")
	if err := os.WriteFile(file, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if owner, err := enginefiles.OwnerSID(file); err != nil || owner != sid {
		t.Fatalf("new cache file owner %s (%v), want %s", owner, err, sid)
	}
	// Everyone gains the same full control: the ACL is no longer private.
	setDACL(t, folder, privateSDDL(sid)+"(A;OICI;FA;;;WD)", windows.PROTECTED_DACL_SECURITY_INFORMATION)
	if ok, err := folderPrivate(folder, sid); err != nil || ok {
		t.Fatalf("an ACL granting Everyone counted as private: %v %v", ok, err)
	}
	// The three private entries with inheritance from the parent turned back
	// on. The parent grants nothing inheritable, so only the protection bit
	// differs from the private ACL.
	parent := filepath.Dir(folder)
	setDACL(t, parent, "D:P(A;;FA;;;"+sid+")", windows.PROTECTED_DACL_SECURITY_INFORMATION)
	defer setDACL(t, parent, "D:P(A;OICI;FA;;;"+sid+")", windows.PROTECTED_DACL_SECURITY_INFORMATION)
	setDACL(t, folder, privateSDDL(sid), windows.UNPROTECTED_DACL_SECURITY_INFORMATION)
	if ok, err := folderPrivate(folder, sid); err != nil || ok {
		t.Fatalf("an inheriting ACL counted as private: %v %v", ok, err)
	}
}

// setDACL writes the DACL of an SDDL string to a folder with the given
// protection flag.
func setDACL(t *testing.T, folder, sddl string, protection windows.SECURITY_INFORMATION) {
	t.Helper()
	sd, err := windows.SecurityDescriptorFromString(sddl)
	if err != nil {
		t.Fatal(err)
	}
	dacl, _, err := sd.DACL()
	if err != nil {
		t.Fatal(err)
	}
	if err := windows.SetNamedSecurityInfo(folder, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|protection, nil, nil, dacl, nil); err != nil {
		t.Fatal(err)
	}
}

func TestRealCacheGuardEmptiesAfterACLChange(t *testing.T) {
	state := t.TempDir()
	sec := systemCacheSecurity()
	if reason, err := prepareCaches(state, "bundle-a", sec); err != nil || reason != cachesNew {
		t.Fatalf("first preparation: %q %v", reason, err)
	}
	present := plant(t, state)
	if reason, err := prepareCaches(state, "bundle-a", sec); err != nil || reason != "" || !present() {
		t.Fatalf("private caches were emptied: %q %v", reason, err)
	}
	sd, err := windows.SecurityDescriptorFromString("D:(A;OICI;FA;;;WD)")
	if err != nil {
		t.Fatal(err)
	}
	dacl, _, _ := sd.DACL()
	if err := windows.SetNamedSecurityInfo(filepath.Join(cacheRoot(state), "comgr"), windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|windows.UNPROTECTED_DACL_SECURITY_INFORMATION, nil, nil, dacl, nil); err != nil {
		t.Fatal(err)
	}
	if reason, err := prepareCaches(state, "bundle-a", sec); err != nil || reason == "" || present() {
		t.Fatalf("caches survived an ACL opened to Everyone: %q %v", reason, err)
	}
}
