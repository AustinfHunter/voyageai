package voyageai

import (
	"bytes"
	"io"
	"net/http"
	"testing"
)

// trackedBody deliberately does not embed *bytes.Reader: embedding would promote
// its WriteTo method, and io.Copy prefers WriterTo over calling Read, which would
// bypass the drained tracking below.
type trackedBody struct {
	r       *bytes.Reader
	closed  bool
	drained bool
}

func (t *trackedBody) Read(p []byte) (int, error) {
	n, err := t.r.Read(p)
	if err == io.EOF {
		t.drained = true
	}
	return n, err
}

func (t *trackedBody) Close() error {
	t.closed = true
	return nil
}

func newTrackedResponse(status int, body string) (*http.Response, *trackedBody) {
	tb := &trackedBody{r: bytes.NewReader([]byte(body))}
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       tb,
	}, tb
}

func TestHandleResponseClosesBodyOnSuccess(t *testing.T) {
	c := NewClient(&VoyageClientOpts{Key: "APIKEY"})
	resp, tb := newTrackedResponse(200, `{"object":"list","data":[],"model":"m","usage":{"total_tokens":0}}`)

	var respBody EmbeddingResponse
	cont, _, err := c.handleResponse(resp, &respBody)
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	}
	if cont {
		t.Error("expected no retry on success")
	}
	if !tb.closed {
		t.Error("expected response body to be closed")
	}
	if !tb.drained {
		t.Error("expected response body to be fully drained")
	}
}

func TestHandleResponseClosesBodyOn5xx(t *testing.T) {
	c := NewClient(&VoyageClientOpts{Key: "APIKEY"})
	resp, tb := newTrackedResponse(500, "internal error")

	var respBody EmbeddingResponse
	cont, _, err := c.handleResponse(resp, &respBody)
	if err == nil {
		t.Fatal("expected an error on 5xx")
	}
	if !cont {
		t.Error("expected 5xx to be retryable")
	}
	if !tb.closed {
		t.Error("expected response body to be closed on 5xx")
	}
	if !tb.drained {
		t.Error("expected response body to be fully drained on 5xx")
	}
}
