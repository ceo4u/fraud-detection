$ErrorActionPreference = "Stop"
$env:JAVA_HOME = "C:\Users\DT\tools\jdk-17.0.20.1+1"
$env:Path = "C:\Users\DT\tools\jdk-17.0.20.1+1\bin;C:\Users\DT\tools\apache-maven-3.9.6\bin;" + $env:Path

Set-Location "c:\Users\DT\Downloads\health detection\fraudguard-java\frontend"
Write-Host "Building Spring Boot frontend..."
& "C:\Users\DT\tools\apache-maven-3.9.6\bin\mvn.cmd" clean package -DskipTests
