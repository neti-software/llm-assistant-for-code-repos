#!/usr/bin/env python3
"""
Script to download all repositories from a specific GitHub organization.
"""

import os
import sys
import subprocess
import argparse
from typing import List, Dict, Optional
import requests


class GitHubOrgDownloader:
    def __init__(self, organization: str, token: Optional[str] = None, download_dir: str = "./repos"):
        """
        Initialize the GitHub organization downloader.
        
        Args:
            organization: GitHub organization name
            token: GitHub personal access token (optional, but recommended for rate limits)
            download_dir: Directory to download repositories to
        """
        self.organization = organization
        self.token = token
        self.download_dir = download_dir
        self.base_url = "https://api.github.com"
        
        # Setup headers for API requests
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Org-Downloader"
        }
        
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    def get_repositories(self) -> List[Dict]:
        """
        Fetch all repositories from the organization.
        
        Returns:
            List of repository dictionaries
        """
        repositories = []
        page = 1
        per_page = 100
        
        print(f"Fetching repositories for organization: {self.organization}")
        
        while True:
            url = f"{self.base_url}/orgs/{self.organization}/repos"
            params = {
                "page": page,
                "per_page": per_page,
                "type": "all"  # Include all types of repositories
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                
                repos_page = response.json()
                
                if not repos_page:
                    break
                
                repositories.extend(repos_page)
                print(f"Fetched page {page} - {len(repos_page)} repositories")
                page += 1
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching repositories: {e}")
                sys.exit(1)
        
        print(f"Total repositories found: {len(repositories)}")
        return repositories
    
    def clone_repository(self, repo: Dict) -> bool:
        """
        Clone a single repository.
        
        Args:
            repo: Repository dictionary from GitHub API
            
        Returns:
            True if successful, False otherwise
        """
        repo_name = repo["name"]
        clone_url = repo["clone_url"]
        repo_path = os.path.join(self.download_dir, repo_name)
        
        # Skip if repository already exists
        if os.path.exists(repo_path):
            print(f"Repository {repo_name} already exists, skipping...")
            return True
        
        print(f"Cloning {repo_name}...")
        
        try:
            # Use git clone command
            cmd = ["git", "clone", clone_url, repo_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Successfully cloned {repo_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to clone {repo_name}: {e.stderr}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error cloning {repo_name}: {e}")
            return False
    
    def download_all_repositories(self, include_forks: bool = False, include_archived: bool = False):
        """
        Download all repositories from the organization.
        
        Args:
            include_forks: Whether to include forked repositories
            include_archived: Whether to include archived repositories
        """
        # Create download directory if it doesn't exist
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Get all repositories
        repositories = self.get_repositories()
        
        # Filter repositories based on options
        filtered_repos = []
        for repo in repositories:
            if not include_forks and repo["fork"]:
                continue
            if not include_archived and repo["archived"]:
                continue
            filtered_repos.append(repo)
        
        print(f"\nFiltered repositories to download: {len(filtered_repos)}")
        
        if not filtered_repos:
            print("No repositories to download.")
            return
        
        # Download repositories
        successful = 0
        failed = 0
        
        for i, repo in enumerate(filtered_repos, 1):
            print(f"\n[{i}/{len(filtered_repos)}] ", end="")
            
            if self.clone_repository(repo):
                successful += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n{'='*50}")
        print(f"Download Summary:")
        print(f"Total repositories: {len(filtered_repos)}")
        print(f"Successfully downloaded: {successful}")
        print(f"Failed: {failed}")
        print(f"Download directory: {os.path.abspath(self.download_dir)}")
        print(f"{'='*50}")
    
    def list_repositories(self, include_forks: bool = False, include_archived: bool = False):
        """
        List all repositories from the organization without downloading.
        
        Args:
            include_forks: Whether to include forked repositories
            include_archived: Whether to include archived repositories
        """
        repositories = self.get_repositories()
        
        print(f"\nRepositories in {self.organization}:")
        print("-" * 80)
        
        for repo in repositories:
            if not include_forks and repo["fork"]:
                continue
            if not include_archived and repo["archived"]:
                continue
            
            status_flags = []
            if repo["fork"]:
                status_flags.append("FORK")
            if repo["archived"]:
                status_flags.append("ARCHIVED")
            if repo["private"]:
                status_flags.append("PRIVATE")
            
            status = f" [{', '.join(status_flags)}]" if status_flags else ""
            
            print(f"• {repo['name']}{status}")
            print(f"  {repo['description'] or 'No description'}")
            print(f"  {repo['html_url']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Download all repositories from a GitHub organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all repos from 'myorg' to './repos' directory
  python download_org_repos.py myorg
  
  # Download with GitHub token for higher rate limits
  python download_org_repos.py myorg --token YOUR_GITHUB_TOKEN
  
  # Download to specific directory, including forks and archived repos
  python download_org_repos.py myorg --dir /path/to/repos --include-forks --include-archived
  
  # Just list repositories without downloading
  python download_org_repos.py myorg --list-only
        """
    )
    
    parser.add_argument("organization", help="GitHub organization name")
    parser.add_argument("--token", help="GitHub personal access token (recommended)")
    parser.add_argument("--dir", default="./repos", help="Directory to download repositories (default: ./repos)")
    parser.add_argument("--include-forks", action="store_true", help="Include forked repositories")
    parser.add_argument("--include-archived", action="store_true", help="Include archived repositories")
    parser.add_argument("--list-only", action="store_true", help="Only list repositories, don't download")
    
    args = parser.parse_args()
    
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: git is not installed or not available in PATH")
        sys.exit(1)
    
    # Initialize downloader
    downloader = GitHubOrgDownloader(
        organization=args.organization,
        token=args.token,
        download_dir=args.dir
    )
    
    try:
        if args.list_only:
            downloader.list_repositories(
                include_forks=args.include_forks,
                include_archived=args.include_archived
            )
        else:
            downloader.download_all_repositories(
                include_forks=args.include_forks,
                include_archived=args.include_archived
            )
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 