$$ErrorActionPreference = 'Stop'
$$toolsDir = Split-Path -Parent $$MyInvocation.MyCommand.Definition

# the release's zip, checked against its SHA-256; Chocolatey shims urna.exe.
Install-ChocolateyZipPackage -PackageName 'urna' -UnzipLocation $$toolsDir `
  -Url64bit '${url_x86_64_windows}' `
  -Checksum64 '${sha256_x86_64_windows}' -ChecksumType64 'sha256'
