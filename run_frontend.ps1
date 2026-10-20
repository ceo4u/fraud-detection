$ErrorActionPreference = "Stop"
$env:JAVA_HOME = "C:\Users\DT\tools\jdk-17.0.20.1+1"
$env:Path = "C:\Users\DT\tools\jdk-17.0.20.1+1\bin;" + $env:Path

Set-Location "c:\Users\DT\Downloads\health detection\fraudguard-java\frontend"
& "C:\Users\DT\tools\jdk-17.0.20.1+1\bin\java.exe" -jar target/fraudguard-1.0.jar
