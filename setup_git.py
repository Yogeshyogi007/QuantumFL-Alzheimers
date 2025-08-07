import subprocess
import os
import sys

def run_command(command):
    """Run a command and return the output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def main():
    print("Setting up Git repository and pushing to GitHub...")
    
    # Step 1: Configure git to not use pager
    print("1. Configuring git...")
    run_command('git config --global core.pager ""')
    run_command('git config --global init.defaultBranch main')
    
    # Step 2: Remove archive folder from tracking
    print("2. Removing archive folder from tracking...")
    run_command('git rm -r --cached archive')
    
    # Step 3: Add all files except those in .gitignore
    print("3. Adding files to git...")
    run_command('git add .')
    
    # Step 4: Commit
    print("4. Committing changes...")
    run_command('git commit -m "Initial commit: QuantumFL-Alzheimers project"')
    
    # Step 5: Rename branch to main
    print("5. Setting branch to main...")
    run_command('git branch -M main')
    
    # Step 6: Add remote
    print("6. Adding remote repository...")
    run_command('git remote add origin https://github.com/Yogeshyogi007/QuantumFL-Alzheimers.git')
    
    # Step 7: Push to GitHub
    print("7. Pushing to GitHub...")
    code, stdout, stderr = run_command('git push -u origin main')
    
    if code == 0:
        print("✅ Successfully pushed to GitHub!")
        print("Repository: https://github.com/Yogeshyogi007/QuantumFL-Alzheimers")
    else:
        print("❌ Error pushing to GitHub:")
        print(f"Error: {stderr}")
        print("Please check your GitHub credentials and try again.")

if __name__ == "__main__":
    main()
