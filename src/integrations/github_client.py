# src/integrations/github_client.py
import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from datetime import datetime, timezone   


import asyncio                       # ← already there

from config import settings

_GH_SEMAPHORE = asyncio.Semaphore(settings.MAX_API_CONCURRENCY or 8)  # NEW



class GitHubClient:
    """Client for fetching developer data from GitHub API."""
    
    def __init__(self):
        # ───────────────────────────────────────────────────────────
        # Base URL and personal-access-token (PAT) selection
        # ───────────────────────────────────────────────────────────
        self.base_url = "https://api.github.com"

        # Round-robin pick from the PAT pool (or fallback to single token)
        pat: str = settings.pick_github_token()          # ← NEW
        masked_pat = f"…{pat[-4:]}" if pat else "None"
        print(f"🔍 DEBUG: GitHubClient picked PAT idx {settings._token_idx} — {masked_pat}")

        # ───────────────────────────────────────────────────────────
        # Standard request headers
        # ───────────────────────────────────────────────────────────
        self.headers: Dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Task-Router/1.0"
        }
        if pat:                                   # add auth header only if PAT present
            self.headers["Authorization"] = f"token {pat}"

        # ───────────────────────────────────────────────────────────
        # Concurrency semaphore shared across all GitHub requests
        # ───────────────────────────────────────────────────────────
        self._sem = asyncio.Semaphore(settings.MAX_API_CONCURRENCY)

    
    def _headers(self) -> Dict[str, str]:
        """
        Rotate through PAT-tokens each request (round-robin).
        Falls back to settings.GITHUB_TOKEN when the pool is empty.
        """
        token = settings.pick_github_token()
        h = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Task-Router/1.0",
        }
        if token:
            h["Authorization"] = f"token {token}"
        return h

    async def _throttle(self, mult: float = 1.0) -> None:
        """Polite sleep helper so we stay well under secondary rate-limits."""
        await asyncio.sleep(0.1 * mult)


    async def get_developer_data(self, username: str, days_back: int = 180) -> Dict:
        """Fetch comprehensive developer data from GitHub."""
        
        print(f"🔍 DEBUG: Fetching developer data for {username}")
        
        async with _GH_SEMAPHORE, aiohttp.ClientSession(headers=self._headers()) as session:
            # Get basic user info
            user_info = await self._get_user_info(session, username)
            print(f"🔍 DEBUG: User info for {username}: {user_info}")
            
            # Get repositories
            repos = await self._get_user_repositories(session, username)
            print(f"🔍 DEBUG: Repositories for {username}: {len(repos)} found")
            
            # Get commits across repositories
            commits = await self._get_user_commits(session, username, repos, days_back)
            print(f"🔍 DEBUG: Commits for {username}: {len(commits)} found")
            
            # after we collect commits
            if not commits and days_back < 180:
                print(f"👀  {username}: no commits in last {days_back}d, widening window to 180d")
                commits = await self._get_user_commits(session, username, repos, 180)
            
            # Get pull requests and reviews
            pr_data = await self._get_pull_request_data(session, username, repos, days_back)
            print(f"🔍 DEBUG: PR data for {username}: {len(pr_data.get('reviews', []))} reviews, {len(pr_data.get('descriptions', []))} descriptions")
            
            # Get issue comments
            issue_comments = await self._get_issue_comments(session, username, repos, days_back)
            print(f"🔍 DEBUG: Issue comments for {username}: {len(issue_comments)} found")
            
            result = {
                "developer_id": username,
                "github_username": username,
                "name": user_info.get("name"),
                "email": user_info.get("email"),
                "commits": commits,
                "pr_reviews": pr_data.get("reviews", []),
                "pr_descriptions": pr_data.get("descriptions", []),
                "issue_comments": issue_comments,
                "discussions": [],  # GitHub Discussions API requires separate implementation
                "commit_messages": [commit.get("message", "") for commit in commits]
            }
            
            print(f"🔍 DEBUG: Final developer data summary for {username}:")
            print(f"  - Commits: {len(result['commits'])}")
            print(f"  - PR Reviews: {len(result['pr_reviews'])}")
            print(f"  - Issue Comments: {len(result['issue_comments'])}")
            
            return result
    
    async def _get_user_info(self, session: aiohttp.ClientSession, username: str) -> Dict:
        """Get basic user information."""
        
        try:
            async with session.get(f"{self.base_url}/users/{username}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching user info: {response.status}")
                    return {}
        except Exception as e:
            print(f"Error fetching user info: {e}")
            return {}
    
    async def _get_user_repositories(self, session: aiohttp.ClientSession, 
                                   username: str, max_repos: int = 50) -> List[Dict]:
        """Get user's repositories."""
        
        try:
            repos = []
            page = 1
            
            while len(repos) < max_repos:
                url = f"{self.base_url}/users/{username}/repos"
                params = {
                    "per_page": min(100, max_repos - len(repos)),
                    "page": page,
                    "sort": "updated",
                    "direction": "desc"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        page_repos = await response.json()
                        if not page_repos:
                            break
                        repos.extend(page_repos)
                        page += 1
                    else:
                        break
            
            return repos[:max_repos]
            
        except Exception as e:
            print(f"Error fetching repositories: {e}")
            return []
    
    async def _get_user_commits(self, session: aiohttp.ClientSession,
                              username: str, repos: List[Dict], 
                              days_back: int) -> List[Dict]:
        """Get user's commits across repositories."""
        
        since_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        all_commits = []
        
        # Limit to most active repositories to avoid rate limits
        active_repos = sorted(repos, key=lambda r: r.get("updated_at", ""), reverse=True)[:10]
        
        for repo in active_repos:
            try:
                repo_commits = await self._get_repo_commits(
                    session, repo["owner"]["login"], repo["name"], username, since_date
                )
                all_commits.extend(repo_commits)
                
                # Rate limiting
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"Error fetching commits for {repo['name']}: {e}")
                continue
        
        return all_commits
    
    async def _get_repo_commits(self, session: aiohttp.ClientSession,
                              owner: str, repo: str, username: str, since: str) -> List[Dict]:
        """Get commits from a specific repository."""
        
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/commits"
            params = {
                "author": username,
                "since": since,
                "per_page": 100
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    commits = await response.json()
                    
                    # Enrich commits with file information
                    enriched_commits = []
                    for commit in commits[:20]:  # Limit detailed analysis
                        detailed_commit = await self._get_commit_details(
                            session, owner, repo, commit["sha"]
                        )
                        if detailed_commit:
                            enriched_commits.append(detailed_commit)
                        
                        await asyncio.sleep(0.05)  # Rate limiting
                    
                    return enriched_commits
                else:
                    return []
                    
        except Exception as e:
            print(f"Error fetching repo commits: {e}")
            return []
    
    async def _get_commit_details(self, session: aiohttp.ClientSession,
                                owner: str, repo: str, sha: str) -> Optional[Dict]:
        """Get detailed commit information including files changed."""
        
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/commits/{sha}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    commit_data = await response.json()
                    
                    return {
                        "hash": sha,
                        "message": commit_data["commit"]["message"],
                        "timestamp": commit_data["commit"]["committer"]["date"],
                        "files": commit_data.get("files", []),
                        "additions": commit_data["stats"]["additions"],
                        "deletions": commit_data["stats"]["deletions"]
                    }
                else:
                    return None
                    
        except Exception as e:
            print(f"Error fetching commit details: {e}")
            return None
    
    async def _get_pull_request_data(self, session: aiohttp.ClientSession,
                                   username: str, repos: List[Dict], 
                                   days_back: int) -> Dict:
        """Get pull request reviews and descriptions."""
        
        since_date = datetime.utcnow() - timedelta(days=days_back)
        reviews = []
        descriptions = []
        
        # Focus on most active repos
        active_repos = repos[:5]
        
        for repo in active_repos:
            try:
                repo_prs = await self._get_repo_pull_requests(
                    session, repo["owner"]["login"], repo["name"], since_date
                )
                
                for pr in repo_prs:
                    # Get PR reviews by this user
                    pr_reviews = await self._get_pr_reviews(
                        session, repo["owner"]["login"], repo["name"], pr["number"], username
                    )
                    reviews.extend(pr_reviews)
                    
                    # Collect PR descriptions if authored by user
                    if pr["user"]["login"] == username:
                        descriptions.append({
                            "description": pr.get("body", ""),
                            "title": pr.get("title", ""),
                            "created_at": pr.get("created_at")
                        })
                    
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching PR data for {repo['name']}: {e}")
                continue
        
        return {
            "reviews": reviews,
            "descriptions": descriptions
        }
   
    async def _get_repo_pull_requests(
        self, session: aiohttp.ClientSession,
        owner: str, repo: str, since_date: datetime
    ) -> List[Dict]:
        """
        Return PRs updated ≥ since_date, handling timezone-aware strings safely.
        """
        # Normalise since_date → UTC & offset-aware
        if since_date.tzinfo is None:
            since_date = since_date.replace(tzinfo=timezone.utc)

        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {
            "state": "all",
            "sort": "updated",
            "direction": "desc",
            "per_page": 50,
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []

                prs = await resp.json()
                recent_prs: List[Dict] = []
                for pr in prs:
                    # GitHub returns e.g. "2025-06-26T12:34:56Z"
                    updated_at = datetime.fromisoformat(
                        pr["updated_at"].replace("Z", "+00:00")
                    )
                    if updated_at >= since_date:
                        recent_prs.append(pr)
                    else:
                        break         # list is already sorted desc

                return recent_prs
        except Exception as exc:
            print(f"Error fetching pull requests for {owner}/{repo}: {exc}")
            return []

    
    async def _get_pr_reviews(self, session: aiohttp.ClientSession,
                            owner: str, repo: str, pr_number: int, username: str) -> List[Dict]:
        """Get reviews for a specific pull request by user."""
        
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
            
            async with session.get(url) as response:
                if response.status == 200:
                    all_reviews = await response.json()
                    
                    # Filter reviews by user
                    user_reviews = [
                        {
                            "content": review.get("body", ""),
                            "state": review.get("state"),
                            "submitted_at": review.get("submitted_at"),
                            "pr_number": pr_number
                        }
                        for review in all_reviews
                        if review["user"]["login"] == username and review.get("body")
                    ]
                    
                    return user_reviews
                else:
                    return []
                    
        except Exception as e:
            print(f"Error fetching PR reviews: {e}")
            return []
    
    async def _get_issue_comments(self, session: aiohttp.ClientSession,
                                username: str, repos: List[Dict], 
                                days_back: int) -> List[Dict]:
        """Get issue comments by user."""
        
        since_date = datetime.utcnow() - timedelta(days=days_back)
        all_comments = []
        
        # Focus on most active repos
        active_repos = repos[:5]
        
        for repo in active_repos:
            try:
                # Get issues with comments
                issues = await self._get_repo_issues(
                    session, repo["owner"]["login"], repo["name"], since_date
                )
                
                for issue in issues:
                    comments = await self._get_issue_comments_for_issue(
                        session, repo["owner"]["login"], repo["name"], 
                        issue["number"], username, since_date
                    )
                    all_comments.extend(comments)
                    
                    await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching issue comments for {repo['name']}: {e}")
                continue
        
        return all_comments
    
    async def _get_repo_issues(self, session: aiohttp.ClientSession,
                             owner: str, repo: str, since_date: datetime) -> List[Dict]:
        """Get issues from repository."""
        
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            params = {
                "state": "all",
                "sort": "updated",
                "direction": "desc",
                "per_page": 30,
                "since": since_date.isoformat()
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
                    
        except Exception as e:
            print(f"Error fetching issues: {e}")
            return []
    
    async def _get_issue_comments_for_issue(self, session: aiohttp.ClientSession,
                                          owner: str, repo: str, issue_number: int,
                                          username: str, since_date: datetime) -> List[Dict]:
        """Get comments for a specific issue by user."""
        
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"
            
            async with session.get(url) as response:
                if response.status == 200:
                    all_comments = await response.json()
                    
                    # Filter comments by user and date
                    user_comments = []
                    for comment in all_comments:
                        if comment["user"]["login"] == username:
                            created_at = datetime.fromisoformat(comment["created_at"].replace("Z", "+00:00"))
                            if created_at >= since_date:
                                user_comments.append({
                                    "content": comment.get("body", ""),
                                    "created_at": comment.get("created_at"),
                                    "issue_number": issue_number
                                })
                    
                    return user_comments
                else:
                    return []
                    
        except Exception as e:
            print(f"Error fetching issue comments: {e}")
            return []
    
    async def get_repository_issues(self, owner: str, repo: str, 
                              state: str = "open", labels: List[str] = None,
                              max_issues: int = 50) -> List[Dict]:
        """Fetch issues from a GitHub repository."""
        
        async with _GH_SEMAPHORE, aiohttp.ClientSession(headers=self._headers()) as session:
            issues = []
            page = 1
            
            while len(issues) < max_issues:
                params = {
                    "state": state,
                    "per_page": min(100, max_issues - len(issues)),
                    "page": page,
                    "sort": "updated",
                    "direction": "desc"
                }
                
                if labels:
                    params["labels"] = ",".join(labels)
                
                try:
                    url = f"{self.base_url}/repos/{owner}/{repo}/issues"
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            page_issues = await response.json()
                            if not page_issues:
                                break
                            
                            # Filter out pull requests (they appear in issues API)
                            actual_issues = [issue for issue in page_issues if 'pull_request' not in issue]
                            issues.extend(actual_issues)
                            page += 1
                            
                            await asyncio.sleep(0.1)  # Rate limiting
                        else:
                            print(f"Error fetching issues: {response.status}")
                            break
                            
                except Exception as e:
                    print(f"Error fetching issues: {e}")
                    break
            
            return issues[:max_issues]

    async def get_issue_details(self, owner: str, repo: str, issue_number: int) -> Optional[Dict]:
        """Get detailed information about a specific issue."""
        
        async with _GH_SEMAPHORE, aiohttp.ClientSession(headers=self._headers()) as session:
            try:
                url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        issue_data = await response.json()
                        
                        # Get comments
                        comments = await self._get_issue_comments_detailed(
                            session, owner, repo, issue_number
                        )
                        issue_data['comments_data'] = comments
                        
                        return issue_data
                    else:
                        print(f"Error fetching issue details: {response.status}")
                        return None
                        
            except Exception as e:
                print(f"Error fetching issue details: {e}")
                return None

    async def _get_issue_comments_detailed(self, session: aiohttp.ClientSession,
                                         owner: str, repo: str, issue_number: int) -> List[Dict]:
        """Get detailed comments for an issue."""
        
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
                    
        except Exception as e:
            print(f"Error fetching issue comments: {e}")
            return []

    async def search_repositories(self, query: str, max_repos: int = 10) -> List[Dict]:
        """Search for repositories using GitHub search API."""
        
        async with _GH_SEMAPHORE, aiohttp.ClientSession(headers=self._headers()) as session:
            try:
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(max_repos, 100)
                }
                
                url = f"{self.base_url}/search/repositories"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('items', [])
                    else:
                        print(f"Error searching repositories: {response.status}")
                        return []
                        
            except Exception as e:
                print(f"Error searching repositories: {e}")
                return []
    
    async def get_repository_contributors(self, owner: str, repo: str, 
                                max_contributors: int = 30) -> List[Dict]:
        """Get repository contributors."""
        
        print(f"🔍 DEBUG: Fetching contributors for {owner}/{repo}")
        print(f"🔍 DEBUG: GitHub token configured: {bool(settings.GITHUB_TOKEN)}")
        print(f"🔍 DEBUG: Headers: {self.headers}")
        
        async with _GH_SEMAPHORE, aiohttp.ClientSession(headers=self._headers()) as session:
            try:
                url = f"{self.base_url}/repos/{owner}/{repo}/contributors"
                params = {
                    "per_page": min(max_contributors, 100),
                    "anon": "false"
                }
                
                print(f"🔍 DEBUG: Making request to: {url}")
                print(f"🔍 DEBUG: Params: {params}")
                
                async with session.get(url, params=params) as response:
                    print(f"🔍 DEBUG: Response status: {response.status}")
                    print(f"🔍 DEBUG: Response headers: {dict(response.headers)}")
                    print(f"🔍 DEBUG: Rate limit remaining: {response.headers.get('X-RateLimit-Remaining')}")
                    
                    if response.status == 200:
                        contributors = await response.json()
                        print(f"🔍 DEBUG: Contributors found: {len(contributors)}")
                        if contributors:
                            print(f"🔍 DEBUG: First contributor sample: {contributors[0]}")
                        return contributors
                    else:
                        error_text = await response.text()
                        print(f"🔍 DEBUG: Error response: {error_text}")
                        return []
                        
            except Exception as e:
                print(f"🔍 DEBUG: Exception in get_repository_contributors: {e}")
                return []