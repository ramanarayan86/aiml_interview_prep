# verify-equations

Verify all LaTeX equations in a question file are numerically correct.

## Usage
/verify-equations <file-path>

Example: /verify-equations ml-interview-prep/questions/section-03-pretraining-and-scaling-laws/q08-chinchilla-derivation.md

## What to do

1. Read the file and extract every display math block ($$...$$) and inline math ($...$) that contains a numerical claim
2. Write a Python script that computes each claimed value independently
3. Run the script and compare output to what the file states
4. Report:
   - ✅ CORRECT — value matches
   - ✗ ERROR — actual value, what file claims, percentage difference
5. If errors found, fix them in the file
6. Commit fixes with message: "Fix §N-QNN: correct <description of error>"

Never add Co-Authored-By to commit messages.
