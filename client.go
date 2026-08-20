package voyageai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// A client for the Voyage AI API.
type VoyageClient struct {
	apikey  string
	client  *http.Client
	opts    *VoyageClientOpts
	baseURL string
}

// Optional arguments for the client configuration.
type VoyageClientOpts struct {
	Key        string // A Voyage AI API key
	TimeOut    int    // The timeout for all client requests, in milliseconds. No timeout is set by default.
	MaxRetries int    // The maximum number of retries. Requests will not be retried by default.
	BaseURL    string // The BaseURL for the API. Defaults to the Voyage AI API but can be changed for testing and/or mocking.
}

// Returns a pointer to the given input. Useful when creating [EmbeddingRequestOpts], [MultimodalRequestOpts], and [RerankRequestOpts] literals.
func Opt[T any](opt T) *T {
	return &opt
}

// Returns a new instance of [VoyageClient]
func NewClient(opts *VoyageClientOpts) *VoyageClient {
	client := &http.Client{}
	if opts == nil {
		opts = &VoyageClientOpts{}
	}

	if opts.TimeOut != 0.0 {
		client.Timeout = time.Duration(opts.TimeOut) * time.Millisecond
	}

	baseURL := "https://api.voyageai.com/v1"
	if opts.BaseURL != "" {
		baseURL = opts.BaseURL
	}

	if opts.Key == "" {
		return &VoyageClient{
			apikey:  os.Getenv("VOYAGE_API_KEY"),
			client:  client,
			baseURL: baseURL,
			opts:    opts,
		}
	}

	return &VoyageClient{
		apikey:  opts.Key,
		client:  client,
		baseURL: baseURL,
		opts:    opts,
	}
}

func (c *VoyageClient) do(req *http.Request) (*http.Response, error) {
	req.Header.Set("Authorization", "BEARER "+c.apikey)
	return c.client.Do(req)
}

// handleAPIError returns true if the given error is recoverable and false otherwise.
// The request retry loop will continue if the error is recoverable and it will abort otherwise.
func (c *VoyageClient) handleAPIError(resp *http.Response) (bool, error) {
	code := resp.StatusCode
	var apiError APIError
	if code >= 400 && code < 500 {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return false, err
		}
		err = json.Unmarshal(body, &apiError)
		if err != nil {
			return false, err
		}
	}

	switch code {
	case 400:
		return false, fmt.Errorf("voyage: bad request, detail: %s", apiError.Detail)
	case 401:
		return false, fmt.Errorf("voyage: unauthorized, detail: %s", apiError.Detail)
	case 422:
		return false, fmt.Errorf("voyage: Malformed Request, detail: %s", apiError.Detail)
	case 429:
		return true, fmt.Errorf("voyage: Rate Limit Reached, detail: %s", apiError.Detail)
	default:
		return true, fmt.Errorf("voyage: Server Error")
	}
}

func (c *VoyageClient) handleAPIRequest(reqBody any, respBody any, url string) error {
	if c.opts.MaxRetries == 0 {
		c.opts.MaxRetries = 1
	}

	var lastErr error
	for range c.opts.MaxRetries {
		bb, err := json.Marshal(reqBody)
		if err != nil {
			return err
		}

		req, err := http.NewRequest("POST", url, bytes.NewBuffer(bb))
		if err != nil {
			return err
		}

		resp, err := c.do(req)
		if err != nil {
			return err
		}

		if resp.StatusCode >= 400 {
			cont, err := c.handleAPIError(resp)
			if !cont {
				return err
			}
			lastErr = err
			continue
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}

		err = json.Unmarshal(body, respBody)
		if err != nil {
			return err
		}

		return nil
	}

	return lastErr
}

// Returns a pointer to an [EmbeddingResponse] or an error if the request failed.
//
// Parameters:
//   - texts - A list of texts as a list of strings, such as ["I like cats", "I also like dogs"]
//   - model - Name of the model. Recommended options: voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-code-3, voyage-finance-2, voyage-law-2.
//   - opts - optional parameters, see [EmbeddingRequestOpts]
func (c *VoyageClient) Embed(texts []string, model string, opts *EmbeddingRequestOpts) (*EmbeddingResponse, error) {
	var reqBody EmbeddingRequest
	var respBody EmbeddingResponse
	if opts != nil {
		reqBody = EmbeddingRequest{
			Input:           texts,
			Model:           model,
			InputType:       opts.InputType,
			Truncation:      opts.Truncation,
			OutputDimension: opts.OutputDimension,
			OutputDType:     opts.OutputDType,
			EncodingFormat:  opts.EncodingFormat,
		}
	} else {
		reqBody = EmbeddingRequest{
			Input: texts,
			Model: model,
		}
	}

	err := c.handleAPIRequest(&reqBody, &respBody, c.baseURL+"/embeddings")
	return &respBody, err
}

// Returns a pointer to an [EmbeddingResponse] or an error if the request failed.
//
// Parameters:
//   - inputs - A list of multimodal inputs to be vectorized. See the "[Voyage AI docs]" for info on constraints.
//   - model - Name of the model. Recommended options: voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-code-3, voyage-finance-2, voyage-law-2.
//   - opts - Optional parameters, see [MultimodalRequestOpts]
//
// [Voyage AI docs]: https://docs.voyageai.com/docs/multimodal-embeddings
func (c *VoyageClient) MultimodalEmbed(inputs []MultimodalContent, model string, opts *MultimodalRequestOpts) (*EmbeddingResponse, error) {
	var reqBody MultimodalRequest
	var respBody EmbeddingResponse
	if opts != nil {
		reqBody = MultimodalRequest{
			Inputs:        inputs,
			Model:         model,
			InputType:     opts.InputType,
			Truncation:    opts.Truncation,
			OuputEncoding: opts.OuputEncoding,
		}
	} else {
		reqBody = MultimodalRequest{
			Inputs: inputs,
			Model:  model,
		}
	}

	if c.opts.MaxRetries == 0 {
		c.opts.MaxRetries = 1
	}

	err := c.handleAPIRequest(&reqBody, &respBody, c.baseURL+"/multimodalembeddings")
	return &respBody, err
}

// Returns a pointer to a [RerankResponse] or an error if the request failed.
//
// Parameters:
//   - query - The query as a string.
//     The query can contain a maximum of 4000 tokens for rerank-2, 2000 tokens
//     for rerank-2-lite and rerank-1, and 1000 tokens for rerank-lite-1.
//   - documents -  The documents to be reranked as a list of strings.
//   - model - Name of the model. Recommended options: rerank-2, rerank-2-lite.
//   - opts - Optional parameters, see [RerankRequestOpts]
//
// [Voyage AI docs]: https://docs.voyageai.com/docs/multimodal-embeddings/
func (c *VoyageClient) Rerank(query string, documents []string, model string, opts *RerankRequestOpts) (*RerankResponse, error) {
	var reqBody RerankRequest
	var respBody RerankResponse
	if opts != nil {
		reqBody = RerankRequest{
			Query:           query,
			Documents:       documents,
			Model:           model,
			TopK:            opts.TopK,
			ReturnDocuments: opts.ReturnDocuments,
			Truncation:      opts.Truncation,
		}
	} else {
		reqBody = RerankRequest{
			Query:     query,
			Documents: documents,
			Model:     model,
		}
	}

	if c.opts.MaxRetries == 0 {
		c.opts.MaxRetries = 1
	}

	err := c.handleAPIRequest(&reqBody, &respBody, c.baseURL+"/rerank")
	return &respBody, err
}
