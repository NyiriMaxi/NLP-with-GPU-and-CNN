import subprocess
import sys
import pkg_resources



def install_requirements():
   
    with open('requirements.txt', 'r') as file:
        packages = file.readlines()
    
    
    required_packages = [pkg.strip() for pkg in packages]
    
    
    for package in required_packages:
        try:
            
            dist = pkg_resources.get_distribution(package)
            print(f"{package} is already installed (version: {dist.version})")
        except pkg_resources.DistributionNotFound:
            
            print(f"{package} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])