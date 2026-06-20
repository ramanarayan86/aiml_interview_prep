# section-summary

Print a progress summary of all sections — what's answered, what's scaffolded, what's missing.

## Usage
/section-summary

## What to do

1. For each section directory in ml-interview-prep/questions/:
   - Count files with full answers (qNN-*.md that exist and are >100 lines)
   - Count scaffolded questions (listed in README but no file yet)
   - Report: Section name | Answered | Scaffolded | Total
2. Print a markdown table
3. Highlight sections where Basic questions are unanswered (highest priority)
4. Suggest the next 3 questions to answer based on: Basic first, then Intermediate, by section order
