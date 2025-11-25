"""
CUDAgraveyard utils: Auto-post blogposts to X (Twitter)
"""

import requests
import os
from datetime import datetime

# Assuming X API v2, but hypothetical in 2025
# Replace with real X API when available

def post_blogpost_to_x(blogpost_md, hashtags, kill_name):
    """
    Post blogpost snippet to X account.
    Requires X bearer token.
    """
    bearer_token = os.getenv("X_BEARERacyj_TOKEN")
    if not bearer_token:
        print("No X_BEARER_TOKEN, skipping tweet")
        return

    # Extract snippet from README blogpost
    # Real implementation would use X API
    tweet_text = f"🚀 CUDAgraveyard Kill: {kill_name} slaughtered! {blogpost_md.split('\\n')[0][:200]} {hashtags}"

    url = "https://api.twitter.com/2/tweets"  # Hypothetical
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    data = {
        "text": tweet_text
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("Tweet posted: ", response.json()["data"]["id"])
    else:
        print(f"Tweet failed: {response.text}")

if __name__ == "__main__":
    post_blogpost_to_x("RIP cuBLAS", "#cuBLASKilled", "GEMM")
