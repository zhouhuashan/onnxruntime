# downloads CMake and puts it in the given output directory
Param(
    [Parameter(Mandatory=$True)]
    [string]$outputDirectory
)

$ErrorActionPreference = "Stop"

$cmakeDirectory = $(Join-Path $outputDirectory "cmake")

# create temporary directory
$tempDir = $(Join-Path $pwd "get_cmake.tmp")
New-Item -Path $tempDir -ItemType "directory" | Out-Null

try
{
    # download CMake
    Write-Host "Downloading CMake..."
    $cmakeUrl = "https://cmake.org/files/v3.10/cmake-3.10.0-win64-x64.zip"
    Invoke-WebRequest -Uri $cmakeUrl -OutFile $(Join-Path $tempDir "cmake.zip")

    # extract cmake.zip
    Write-Host "Extracting CMake..."
    Add-Type -Assembly "System.IO.Compression.FileSystem"
    [System.IO.Compression.ZipFile]::ExtractToDirectory(
        $(Join-Path $tempDir "cmake.zip"),
        $(Join-Path $tempDir "cmake"))

    # clean existing directory, if any
    if (Test-Path -LiteralPath $cmakeDirectory -PathType Container)
    {
        Remove-Item -LiteralPath $cmakeDirectory -Recurse -Force
    }

    # move extracted files
    $extractedCmakeDirectory = $(Join-Path $(Join-Path $tempDir "cmake") "cmake*" -Resolve)
    Move-Item $extractedCmakeDirectory $cmakeDirectory
}
finally
{
    # clean up temporary directory
    Remove-Item $tempDir -Recurse
}
