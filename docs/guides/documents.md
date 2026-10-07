# Documents and citations

Documents let beliefs be grounded in exact spans of text, with the span checked by code.

## Registering documents

```python
kb.add_document("10-K", open("acme-10k.txt").read())

# or, with an agent
agent = Agent(model, documents={"10-K": text, "earnings-call": transcript})
```

Registered documents are shown to the model in a `# Documents` section of its context, truncated at
`Projector(max_document_chars=...)` (50,000 characters by default). Truncation is announced in the
prompt.

## Citing

```python
kb.cite(
    "revenue:Q2",
    4.3e9,
    document="10-K",
    quote="Total revenue for the second quarter was $4.3 billion",
    claim="Q2 revenue",
)
```

`cite` creates a premise with source `document:10-K`, after two checks:

1. **The quote appears in the document.** Matching ignores case, collapses whitespace and normalizes curly
   quotes and dashes. Otherwise it is exact: no paraphrases.
2. **The value is stated in the quote** (pass `check_value=False` to skip). Strings must appear in the
   quote as a whole word or phrase. Numbers must match a number in the quote after rounding to its stated
   precision, at the scale its unit states: `4.3e9` matches "$4.3 billion", "$4.3B" and "4,300 million",
   with Portuguese and Spanish scale words such as "milhões", "bilhões", "millones" and "billones" also recognized,
   and `0.0976` matches "9.76%" (or "9.76 percent", "9.76 percentage points", "9.76pp"), but `4.3e9` does
   not match "$4 million". A value stored in a smaller unit matches too: `4300`, in millions, matches
   "$4.3 billion". A number without a unit may be in any magnitude or a percentage, because tables often
   state their unit in a header ("in millions", "(%)"). A numeric string value (`"4.3"`) is matched as a
   number.

The second check is what stops a model from citing a real sentence while extracting a number that isn't
in it.

A model cites with the `cite` action of the [claim contract](contract.md#cite). Failed checks are
rejected and fed back like any other contract violation.

## Verification

`CitationCheck` re-checks every cited premise in a proof against the current document text. If a document
is replaced after a citation was recorded, and the quote no longer appears, verification fails. If the
document isn't available (verifying a proof without its base), the citation produces a warning instead.

## Current limitations

- Documents are plain text. For PDFs or HTML, extract the text first.
- Quotes are matched as substrings. There are no character offsets yet, so a quote that appears twice
  isn't disambiguated.
- Large document sets should be retrieved per task rather than all registered at once. Retrieval
  integration is on the roadmap.
