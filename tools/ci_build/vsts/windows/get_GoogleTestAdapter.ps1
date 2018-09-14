# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# downloads GoogleTestAdapter and puts it in the given output directory
Param(
    [Parameter(Mandatory=$true, HelpMessage="Target output directory to copy GoogleTestAdapter nupkg folder to. Must exist.")][string]$outputDirectory
)

# no version number in the name for simplicity in the CI scripts
$targetDirectory = $(Join-Path $outputDirectory "GoogleTestAdapter")

# create temporary directory
$tempDir = $(Join-Path $pwd "get_GoogleTestAdapter.tmp")
Remove-Item $tempDir -Recurse -Force -ErrorAction Ignore  # remove if an old one exists
New-Item -Path $tempDir -ItemType "directory" | Out-Null

try
{
    # download GoogleTestAdapter
    Write-Host "Downloading GoogleTestAdapter 0.12.3 nupkg..."
    $url = "https://www.nuget.org/api/v2/package/GoogleTestAdapter/0.12.3"
    $targetFile = $(Join-Path $tempDir "GoogleTestAdapter.0.12.3.nupkg")
    Invoke-WebRequest -Uri $url -OutFile $targetFile

    # extract 
    $extractDir = Join-Path $tempDir "GoogleTestAdapter.0.12.3"
    Write-Host "Extracting GoogleTestAdapter from $targetFile to $extractDir..."
    Add-Type -Assembly "System.IO.Compression.FileSystem"
    [System.IO.Compression.ZipFile]::ExtractToDirectory(
        $targetFile,
        $extractDir)

    # clean existing directory, if any
    if (Test-Path -LiteralPath $targetDirectory -PathType Container)
    {
        Remove-Item -LiteralPath $targetDirectory -Recurse -Force
    }

    # move extracted files
    Write-Host "Moving $extractDir to $targetDirectory"
    Move-Item $extractDir $targetDirectory
}
catch
{
    $ErrorMessage = $_.Exception.Message
    Write-Error "Exception acquiring GoogleTestAdapter: Error=$ErrorMessage"
}
finally
{
    # clean up temporary directory
    Write-Host "Removing temporary directory"
    Remove-Item $tempDir -Recurse -Force
}
