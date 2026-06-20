# fix-svgs

Audit and fix all SVG files in the assets directory for GitHub rendering issues.

## Usage
/fix-svgs [section]

Example: /fix-svgs s3   (only Section 3 assets)
Example: /fix-svgs      (all assets)

## What to check and fix

For every SVG file found:

1. **Missing xmlns** — add `xmlns="http://www.w3.org/2000/svg"` to the root `<svg>` tag
2. **Missing viewBox** — add `viewBox="0 0 W H"` where W/H match the width/height attributes
3. **Invalid XML entities** — replace `&nbsp;` with a literal space or `&#160;`
4. **Broken text** — ensure all `<text>` elements have valid UTF-8 content
5. **Unclosed tags** — validate SVG is well-formed XML

Report all issues found and fixed. Commit with: "Fix SVG rendering: <summary>"

Never add Co-Authored-By to commit messages.
