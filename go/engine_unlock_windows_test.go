package thenoise

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-identity"
)

func TestUnlockFinishesPartlyUnlockedEngineAtSameFileAlias(t *testing.T) {
	engine, state, pin := installSmall(t)
	child := filepath.Join(engine, "Lib", "site-packages")
	sid, err := programfiles.AccountSID()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := programfiles.UnlockWithin(engine); err != nil {
			t.Error(err)
		}
	})
	// The parent is already writable; this child retains an installed ACL.
	if err := programfiles.SetFolderACL(child, sid, programfiles.ReadOnly); err != nil {
		t.Fatal(err)
	}
	alias := strings.ToUpper(engine)
	if same, err := identity.SameFile(engine, alias); err != nil || !same {
		t.Fatalf("fixture alias is not the same file: %v %v", same, err)
	}
	if err := unlockEngine(t.Context(), alias, manifestFile(state), pin); err != nil {
		t.Fatal(err)
	}
	if readOnly, err := programfiles.FolderACLIs(child, sid, programfiles.ReadOnly); err != nil || readOnly {
		t.Fatalf("child stayed read-only: %v %v", readOnly, err)
	}
	if err := os.WriteFile(filepath.Join(child, "after-unlock.txt"), []byte("writable"), 0o600); err != nil {
		t.Fatal(err)
	}
}
