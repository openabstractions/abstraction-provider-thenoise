package thenoise

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
)

func TestUnlockFinishesPartlyUnlockedEngineAtSameFileAlias(t *testing.T) {
	engine, state, pin := installSmall(t)
	child := filepath.Join(engine, "Lib", "site-packages")
	sid, err := enginefiles.AccountSID()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := enginefiles.UnlockWithin(engine); err != nil {
			t.Error(err)
		}
	})
	// The parent is already writable; this child retains an installed ACL.
	if err := enginefiles.SetFolderACL(child, sid, enginefiles.ReadOnly); err != nil {
		t.Fatal(err)
	}
	alias := strings.ToUpper(engine)
	if same, err := identity.SameFile(engine, alias); err != nil || !same {
		t.Fatalf("fixture alias is not the same file: %v %v", same, err)
	}
	if err := unlockEngine(t.Context(), alias, manifestFile(state), pin); err != nil {
		t.Fatal(err)
	}
	if readOnly, err := enginefiles.FolderACLIs(child, sid, enginefiles.ReadOnly); err != nil || readOnly {
		t.Fatalf("child stayed read-only: %v %v", readOnly, err)
	}
	if err := os.WriteFile(filepath.Join(child, "after-unlock.txt"), []byte("writable"), 0o600); err != nil {
		t.Fatal(err)
	}
}
