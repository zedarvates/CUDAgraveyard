"""
CUDAgraveyard utils: Auto-keyword feature-php blogpost generator for kills
"""

import json
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_blogpost(task, kill_dir, kill_results):
    """
    Generate arrogant blogpost based on task style directives.
    """
    style = "\n".join(task.get('style_directives', []))
    output_format = task.get('output_format', {})
    length = output_format.get('blogpost.md', '500 mots maximum')

    prompt = f"""
    {style}
    Generate a triumphant blogpost announcing the kill of the baseline: {task.get('baseline')}.
    Results: {kill_results}
    Style: ultra arrogant, technical bragging, ending with hashtags.
    Length: {length}
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=1024
    )

    blogpost = response.content[0].text
    with open(f"{kill_dir}/blogpost.md", "w") as f:
        f.write(blogpost)

if __name__ == "__main__":
    # Test
    with open("../templates/flashattention_toon_v3.json") as f:
        task = json.load(f)
    generate_blogpost(task, "/tmp", {"tokens/s": 15000, "watts": 250})
