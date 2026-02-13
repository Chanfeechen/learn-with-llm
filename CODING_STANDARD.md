# Coding Standard and Rules

This repository follows the Google Python Style Guide as the primary standard. The guide is the source of truth; this document summarizes expectations and clarifies repo-specific rules.

Reference: https://google.github.io/styleguide/pyguide.html

## General Principles
- Prioritize readability over cleverness.
- Write code for maintainers, not just for the interpreter.
- Keep functions and modules small and focused.
- Favor explicitness; avoid implicit side effects.
- Keep naming consistent and descriptive.

## Formatting
- Use 2-space indentation in Python.
- Limit lines to 80 characters.
- Wrap long expressions using parentheses instead of backslashes.
- Use a single blank line to separate logical sections within functions.
- End files with a single newline.

## Naming
- Modules: lowercase with underscores.
- Classes: `CamelCase`.
- Functions and variables: `lower_with_underscores`.
- Constants: `UPPER_WITH_UNDERSCORES`.
- Avoid single-letter names except for trivial loop indices.

## Docstrings and Comments
- All public modules, classes, and functions must have docstrings.
- Use Google-style docstrings with `Args`, `Returns`, and `Raises` when applicable.
- Use complete sentences and proper punctuation in docstrings and comments.
- Comments should explain intent or non-obvious decisions, not restate code.

## Imports
- Use absolute imports whenever possible.
- Order imports: standard library, third-party, then local.
- Avoid wildcard imports.

## Type Hints
- Use type annotations for public APIs and non-trivial functions.
- Keep annotations readable; avoid overly complex nested types.

## Errors and Exceptions
- Raise specific exceptions; avoid bare `except`.
- Use exceptions for exceptional situations, not control flow.

## Testing
- Write tests for new functionality and bug fixes.
- Prefer small, focused tests with clear names.
- Tests should be deterministic and isolated.

## Repo Conventions
- Markdown files should be concise and structured with clear headings.
- Keep examples minimal and runnable when possible.
- If a topic requires extensive detail, split it into multiple files.
