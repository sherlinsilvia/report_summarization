import re

def clean_markdown_and_format_html(text: str, is_dark_bg: bool = False) -> str:
    """
    Parses Markdown syntax like 1. **Heading**: or **text** and converts them into
    clean, styled HTML elements with crisp high-contrast visibility.
    """
    # 1. Remove raw context header leak prefixes if any remain
    text = re.sub(r'\(Page\s+\d+,\s+Section:[^)]+\):\s*', '', text, flags=re.IGNORECASE)
    
    heading_color = "#38bdf8" if is_dark_bg else "#0369a1"
    border_color = "rgba(255,255,255,0.15)" if is_dark_bg else "#cbd5e1"
    bullet_color = "#e2e8f0" if is_dark_bg else "#1e293b"
    
    # 2. Convert numbered headings like "1. **Heading**:" or "**1. Heading**:" into styled section headers
    def replace_heading(match):
        title = match.group(1).strip()
        return f'<h4 style="margin-top: 20px; margin-bottom: 8px; color: {heading_color}; font-family: Outfit, sans-serif; font-size: 1.1rem; font-weight: 700; border-bottom: 1px solid {border_color}; padding-bottom: 4px;">{title}</h4>'
        
    text = re.sub(r'(?:^\d+\.\s*)?\*\*(?:\d+\.\s*)?([^*]+)\*\*:?', replace_heading, text, flags=re.MULTILINE)
    
    # 3. Convert any remaining **text** to <strong>text</strong>
    strong_color = "#38bdf8" if is_dark_bg else "#0284c7"
    text = re.sub(r'\*\*([^*]+)\*\*', rf'<strong style="color: {strong_color};">\1</strong>', text)
    
    # 4. Convert bullet lines starting with - into styled bullets
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        l_str = line.strip()
        if l_str.startswith('- ') or l_str.startswith('• '):
            bullet_text = re.sub(r'^[\-\•]\s*', '', l_str).strip()
            formatted_lines.append(f'<div style="margin-left: 12px; margin-bottom: 6px; color: {bullet_color}; font-size: 0.98rem; line-height: 1.6;">• {bullet_text}</div>')
        elif l_str:
            formatted_lines.append(f'<div style="color: {bullet_color}; margin-bottom: 4px;">{l_str}</div>')
        else:
            formatted_lines.append('')
            
    return '\n'.join(formatted_lines)

def format_summary_citations_html(
    summary_text: str,
    retrieved_chunks: list[dict],
    is_dark_bg: bool = False
) -> tuple[str, list[dict]]:
    """
    Finds brackets like [0], [1] in the summary and replaces them with beautiful
    HTML tooltips/badges.
    Also compiles a list of cited chunks for the reference list.
    """
    # 1. Clean Markdown headers and bold text
    formatted_text = clean_markdown_and_format_html(summary_text, is_dark_bg=is_dark_bg)
    
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
            if retrieved_chunks and idx < len(retrieved_chunks):
                chunk = retrieved_chunks[idx]
                page = chunk.get("page", "?")
                section = chunk.get("section", "General")
                snippet = chunk.get("text", "")
                
                tooltip_snippet = snippet[:150].replace('"', '&quot;').replace("'", "&apos;")
                if len(snippet) > 150:
                    tooltip_snippet += "..."
                    
                badge_html = (
                    f'<span class="citation-badge" '
                    f'style="background-color: #0284c7; color: #ffffff; padding: 2px 7px; '
                    f'border-radius: 4px; font-size: 0.8em; font-weight: 700; '
                    f'cursor: help; margin: 0 3px; display: inline-block; vertical-align: middle; box-shadow: 0 2px 4px rgba(0,0,0,0.2);" '
                    f'title="Page {page} | Section: {section} | {tooltip_snippet}">'
                    f'#{idx}'
                    f'</span>'
                )
                badges.append(badge_html)
            else:
                badge_html = f'<span style="color: #38bdf8; font-weight: bold;">[#{idx}]</span>'
                badges.append(badge_html)
                
        return " " + " ".join(badges) + " "
        
    html_summary = citation_pattern.sub(replace_citation, formatted_text)
    
    # 4. Process the referenced list
    cited_list = []
    if retrieved_chunks:
        for idx in sorted(list(cited_indices)):
            if idx < len(retrieved_chunks):
                chunk = retrieved_chunks[idx]
                cited_list.append({
                    "index": idx,
                    "page": chunk.get("page", "?"),
                    "section": chunk.get("section", "General"),
                    "text": chunk.get("text", "")[:250] + ("..." if len(chunk.get("text", "")) > 250 else "")
                })
                
    return html_summary, cited_list
