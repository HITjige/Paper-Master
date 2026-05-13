Create or update a skill based on the analysis below.

## Rules

### If creating a NEW skill (no existing skill in same domain)
- Use write_file to create `skills/<kebab-case-name>/SKILL.md`
- Before writing, read_file `{{ skill_creator_path }}` for format reference (frontmatter structure, naming conventions)
- Include YAML frontmatter with `name` and `description` fields
- The body should be concise, actionable Markdown covering:
  - **Overview**: 1-2 sentences on what this skill enables
  - **Domain Knowledge**: structured technical insights from the analysis
  - **When to Use**: specific triggers/scenarios
  - **Key References**: paper IDs with brief annotations
  - **Example Queries**: typical questions this skill answers
- Keep under 1500 words

### If UPDATING an existing skill (similar domain already exists)
- Use edit_file to append new insights into the existing skill's SKILL.md
- Add a `## Updates` section with date and new findings if one doesn't exist
- Merge overlapping content — don't duplicate what's already there
- Preserve existing structure; only add genuinely new information

### Deduplication
- Read the existing skills listed below BEFORE creating anything new
- If an existing skill already covers the same topic, edit it instead of creating a new one
- Two skills overlap if they cover: same research area, same methodology family, same application domain

### Quality standards
- Skills teach the agent HOW to answer, not WHAT to answer
- Focus on methodology, comparative insights, and technical depth
- Avoid: paper URLs, raw citations, copy-pasted abstracts

## File paths (relative to workspace root)
- skills/<name>/SKILL.md — for new skills (write_file)
- skills/<name>/SKILL.md — for existing skills (edit_file)

If nothing to update, stop without calling tools.
