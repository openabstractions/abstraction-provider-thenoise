//go:build windows

package thenoise

import "github.com/openabstractions/abstraction-facade/go/programfiles"

// The cache guard uses the shared provider kit's folder ACL: a protected
// DACL granting full control, inherited by files and folders, to the
// account, SYSTEM and Administrators.
func systemCacheSecurity() cacheSecurity {
	return cacheSecurity{account: programfiles.AccountSID, restrict: restrictFolder, private: folderPrivate, owner: programfiles.OwnerSID}
}

func restrictFolder(folder, sid string) error {
	return programfiles.SetFolderACL(folder, sid, programfiles.FullControl)
}

// folderPrivate accepts exactly the ACL restrictFolder writes.
func folderPrivate(folder, sid string) (bool, error) {
	return programfiles.FolderACLIs(folder, sid, programfiles.FullControl)
}
