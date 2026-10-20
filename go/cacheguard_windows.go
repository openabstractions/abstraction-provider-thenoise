//go:build windows

package thenoise

import "github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"

// The cache guard uses the shared provider kit's folder ACL: a protected
// DACL granting full control, inherited by files and folders, to the
// account, SYSTEM and Administrators.
func systemCacheSecurity() cacheSecurity {
	return cacheSecurity{account: enginefiles.AccountSID, restrict: restrictFolder, private: folderPrivate, owner: enginefiles.OwnerSID}
}

func restrictFolder(folder, sid string) error {
	return enginefiles.SetFolderACL(folder, sid, enginefiles.FullControl)
}

// folderPrivate accepts exactly the ACL restrictFolder writes.
func folderPrivate(folder, sid string) (bool, error) {
	return enginefiles.FolderACLIs(folder, sid, enginefiles.FullControl)
}
