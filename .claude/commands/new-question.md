# new-question

Write a complete answer for a new interview question in this repo.

## Usage
/new-question <section-number> <question-number> <slug> "<question text>"

Example: /new-question 3 17 chinchilla-derivation-full "Derive the Chinchilla loss function from scratch"

## What to do

1. Identify the section directory: `ml-interview-prep/questions/section-NN-<name>/`
2. Create the asset directory: `ml-interview-prep/assets/sNqNN/`
3. Write `fig1-*.svg` and `fig2-*.svg` — valid SVGs with xmlns, viewBox, no &nbsp;
4. Write `qNN-<slug>.md` following the 15-section template in CLAUDE.md
5. Update the section `README.md` — flip status from 📝 to ✅, update badge
6. Update `ml-interview-prep/README.md` — update answered count and status tables
7. Verify all LaTeX equations numerically before committing
8. Commit with message: "Add §N-QNN: <short description>"

Never add Co-Authored-By to commit messages.
