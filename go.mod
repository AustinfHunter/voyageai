module github.com/austinfhunter/voyageai

go 1.23.4

retract (
	v0.1.2
	v0.1.1
	v0.1.0
	v1.1.0
	v1.0.0 // handleAPIRequest silently drops retryable errors (429/5xx), returning nil with a zero-value response; see #11
	v1.1.1 // handleAPIRequest silently drops retryable errors (429/5xx), returning nil with a zero-value response; see #11
	v1.1.2 // handleAPIRequest silently drops retryable errors (429/5xx), returning nil with a zero-value response; see #11
	v1.2.0 // handleAPIRequest silently drops retryable errors (429/5xx), returning nil with a zero-value response; see #11
)
