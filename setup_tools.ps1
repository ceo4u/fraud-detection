$ErrorActionPreference = "Stop"
$tools = "C:\Users\DT\tools"
if (!(Test-Path $tools)) {
    New-Item -ItemType Directory -Path $tools | Out-Null
}

$jdkZip = Join-Path $tools "openjdk17.zip"
$mvnZip = Join-Path $tools "maven.zip"

if (!(Test-Path $jdkZip) -or ((Get-Item $jdkZip).Length -lt 10000000)) {
    Write-Host "Downloading OpenJDK 17..."
    curl.exe -L -o $jdkZip "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse"
} else {
    Write-Host "OpenJDK 17 zip already downloaded."
}

if (!(Test-Path $mvnZip) -or ((Get-Item $mvnZip).Length -lt 1000000)) {
    Write-Host "Downloading Maven 3.9.6..."
    curl.exe -L -o $mvnZip "https://archive.apache.org/dist/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.zip"
} else {
    Write-Host "Maven zip already downloaded."
}

Write-Host "Extracting OpenJDK 17..."
tar.exe -xf $jdkZip -C $tools
Write-Host "Extracting Maven..."
tar.exe -xf $mvnZip -C $tools

Write-Host "Toolchain extracted successfully!"
Get-ChildItem $tools
