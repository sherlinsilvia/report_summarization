import re

def clean_markdown_and_format_html(text: str) -> str:
    """
    Parses Markdown syntax like 1. **Heading**: or **text** and converts them into
    clean, styled HTML elements without leaving raw asterisks or bullet dashes.
    """
    # 1. Remove raw context header leak prefixes if any remain
    text = re.sub(r'\(Page\s+\d+,\s+Section:[^)]+\):\s*', '', text, flags=re.IGNORECASE)
    
    # 2. Convert numbered headings like "1. **Heading**:" or "**1. Heading**:" into styled section headers
    def replace_heading(match):
        title = match.group(1).strip()
        return f'<h4 style="margin-top: 18px; margin-bottom: 6px; color: #1e3a8a; font-family: sans-serif; font-size: 1.05rem; font-weight: 700; border-bottom: 1px solid #e2e8f0; padding-bottom: 3px;">{title}</h4>'
        
    # Match patterns like: 1. **Patient Information**: or **1. Patient Information**
    text = re.sub(r'(?:^\d+\.\s*)?\*\*(?:\d+\.\s*)?([^*]+)\*\*:?', replace_heading, text, flags=re.MULTILINE)
    
    # 3. Convert any remaining **text** to <strong>text</strong>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    
    # 4. Convert bullet lines starting with - into styled bullets
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip().startswith('- '):
            bullet_text = line.strip()[2:].strip()
            formatted_lines.append(f'<div style="margin-left: 12px; margin-bottom: 4px; color: #334155;">• {bullet_text}</div>')
        else:
            formatted_lines.append(line)
            
    return '\n'.join(formatted_lines)

def format_summary_citations_html(
    summary_text: str,
    retrieved_chunks: list[dict]
) -> tuple[str, list[dict]]:
    """
    Finds brackets like [0], [1] in the summary and replaces them with beautiful
    HTML tooltips/badges.
    Also compiles a list of cited chunks for the reference list.
    
    Returns:
        - html_summary: str (summary text with inline HTML badges and styled headings)
        - cited_list: list of dicts (only the chunks that were actually cited)
    """
    # 1. Clean Markdown headers and bold text
    formatted_text = clean_markdown_and_format_html(summary_text)
    
    # 2. Find all citation matches
    citation_pattern = re.compile(r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]')
    
    all_matches = citation_pattern.findall(formatted_text)
    cited_indices = set()
    for match in all_matches:
        parts = match.split(',')
        for p in parts:
            try:
                cited_indices.add(int(p.strip()))
            except ValueError:
                pass
                
    # 3. Replace inline brackets with HTML tooltips
    def replace_citation(match):
        match_str = match.group(1)
        indices = [int(idx.strip()) for idx in match_str.split(',') if idx.strip().isdigit()]
        
        badges = []
        for idx in indices:
            if idx < len(retrieved_chunks):
                chunk = retrieved_chunks[idx]
                page = chunk.get("page", "?")
                section = chunk.get("section", "General")
                snippet = chunk.get("text", "")
                
                # Truncate snippet for the tooltip
                tooltip_snippet = snippet[:150].replace('"', '&quot;').replace("'", "&apos;")
                if len(snippet) > 150:
                    tooltip_snippet += "..."
                    
                # Create HTML badge with CSS styling
                badge_html = (
                    f'<span class="citation-badge" '
                    f'style="background-color: #3b82f6; color: white; padding: 1px 6px; '
                    f'border-radius: 4px; font-size: 0.78em; font-weight: bold; '
                    f'cursor: help; margin: 0 2px; display: inline-block; vertical-align: middle;" '
                    f'title="Page {page} | Section: {section} | {tooltip_snippet}">'
                    f'#{idx}'
                    f'</span>'
                )
                badges.append(badge_html)
            else:
                badge_html = f'<span style="color: red; font-weight: bold;">[{idx}]</span>'
                badges.append(badge_html)
                
        return " " + " ".join(badges) + " "
        
    html_summary = citation_pattern.sub(replace_citation, formatted_text)
    
    # 4. Process the referenced list
    cited_list = []
    for idx in sorted(list(cited_indices)):
        if idx < len(retrieved_chunks):
            chunk = retrieved_chunks[idx]
            cited_list.append({
                "index": idx,
                "chunk_id": chunk.get("chunk_id", idx),
                "page": chunk.get("page", "?"),
                "section": chunk.get("section", "General"),
                "text": chunk.get("text", "")
            })
            
    return html_summary, cited_list
