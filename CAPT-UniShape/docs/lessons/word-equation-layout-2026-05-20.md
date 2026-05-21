# Word equation layout recovery

## Root Cause

The chapter-three equation repair initially mixed linear equation syntax, ordinary paragraph text, and symbol explanations in ways that Word could not lay out reliably. This caused visible formula artifacts, non-centered equations, non-right-aligned numbering, and symbol explanations packed into soft line breaks rather than independent paragraphs.

## Conclusion

For Word manuscript equations that require centered formulas and right-aligned numbers, use a borderless three-column table as the layout container: left spacer, centered editable OMML equation, and right-aligned number. Keep every independent formula in its own equation row unless a short group of parallel definitions intentionally shares one number. Convert symbol explanations to real Word paragraphs, not `w:br` soft breaks. Avoid `\mathbb` for tensor dimensions and use `R^{m\times n}` style instead.

Formula explanations should follow journal prose style rather than a glossary list when they appear immediately after an equation. Write one paragraph beginning with `where`, use inline Word equation objects for symbols and dimensions, and move tensor dimensions out of the displayed formula unless the dimension is itself part of the mathematical statement.

## Tags

Word, DOCX, equations, OMML, formula-numbering, symbol-explanations, chapter-3
