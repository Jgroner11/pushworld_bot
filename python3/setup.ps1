# ----------------------------------------------
# How to run this script:
# 1. Open PowerShell and navigate to the python3 directory:
#      cd C:\Users\jacob\OneDrive\Desktop\pushworld_bot\python3
# 2. Run the script:
#      .\setup.ps1
# This script creates a virtual environment, installs requirements,
# and adds the src directory to site-packages via pushworld.pth.
# ----------------------------------------------

# Step 0: Ensure we are in the python3 directory
$projectRoot = Split-Path -Parent (Get-Location)

# Step 1: Create a virtual environment
Write-Host "Creating virtual environment..."
python -m venv "venv"

# Step 2: Activate the virtual environment
Write-Host "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# Step 3: Install the dependencies from requirements.txt (located in parent folder)
Write-Host "Installing dependencies..."
pip install -r "$projectRoot\requirements.txt"

# Step 4: Add python3/src to site-packages via .pth file
Write-Host "Adding src to site-packages..."
$sitePackagesPath = Join-Path (Get-Location) "venv\Lib\site-packages"
$pthFile = Join-Path $sitePackagesPath "pushworld.pth"
"$projectRoot\python3\src" | Out-File -Encoding ASCII -Fil
