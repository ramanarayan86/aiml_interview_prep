# update-status

Sync all README status badges and answered counts to match the actual files on disk.

## Usage
/update-status

## What to do

1. Count ✅ answered questions per section by checking which qNN-*.md files exist
2. Update each section README.md:
   - Badge: `answered-N%2FTotal`
   - Table rows: flip 📝 to ✅ for questions that have answer files
3. Update ml-interview-prep/README.md:
   - Main answered badge
   - "Current status" section text
   - Section status tables (Basic/Intermediate/Advanced/Applied rows)
   - File tree (show answered files with ✅)
4. Update root README.md:
   - Answered badge and table
5. Commit: "Sync README status badges to current answered count"

Never add Co-Authored-By to commit messages.
