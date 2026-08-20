package voyageai_test

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/austinfhunter/voyageai"
)

func TestNewClientNilOpts(t *testing.T) {
	err := os.Setenv("VOYAGE_API_KEY", "dummy_key")
	if err != nil {
		t.Errorf("Couldn't set up environment: %s", err.Error())
	}

	// Make sure there are no dereferences of nil attrs when initialized with environment vars
	_ = voyageai.NewClient(nil)
}

func TestEmbedRequiredArgsResponse(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageai.EmbeddingRequest
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("Could not read request body")
		}

		err = json.Unmarshal(b, &req)
		if err != nil {
			t.Fatalf("Invalid request body")
		}

		if req.Input == nil {
			t.Errorf("Expected non-nil value for 'Input'")
		}

		if req.Model == "" {
			t.Errorf("Expected non-empty value for 'Model'")
		}

		resp := voyageai.EmbeddingResponse{
			Object: "list",
			Data: []voyageai.EmbeddingObject{
				{
					Object:    "embedding",
					Embedding: []float32{0.1, 0.2, 0.3},
					Index:     0,
				},
				{
					Object:    "embedding",
					Embedding: []float32{0.4, 0.5, 0.6},
					Index:     1,
				},
			},
			Model: req.Model,
			Usage: voyageai.UsageObject{
				TotalTokens: 10,
			},
		}

		respb, err := json.Marshal(&resp)
		if err != nil {
			t.Fatal(err.Error())
		}

		w.WriteHeader(201)
		w.Write(respb)
	}))
	defer s.Close()

	cl := voyageai.NewClient(&voyageai.VoyageClientOpts{
		Key:        "APIKEY",
		TimeOut:    1500,
		MaxRetries: 3,
		BaseURL:    s.URL,
	})

	_, err := cl.Embed([]string{"input1", "input2"}, "test-model", nil)
	if err != nil {
		t.Fatal(err.Error())
	}
}

func TestEmbedCustomArgsResponse(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageai.EmbeddingRequest
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("Could not read request body")
		}

		err = json.Unmarshal(b, &req)
		if err != nil {
			t.Fatal("Invalid request body")
		}

		if req.Input == nil {
			t.Errorf("Expected non-nil value for 'Input'")
		}

		if req.Model == "" {
			t.Errorf("Expected non-empty value for 'Model'")
		}

		if req.InputType == nil {
			t.Errorf("Expected non-nil value for 'InputType'")
		}

		if req.Truncation == nil {
			t.Errorf("Expected non-nil value for 'Truncation'")
		}

		if req.EncodingFormat == nil {
			t.Errorf("Expected non-nil value for 'EncodingFormat'")
		}

		if req.OutputDType == nil {
			t.Errorf("Expected non-nil value for 'OutputDType'")
		}

		if req.OutputDimension == nil {
			t.Errorf("Expected non-nil value for 'OutputDimension'")
		}

		resp := voyageai.EmbeddingResponse{
			Object: "list",
			Data: []voyageai.EmbeddingObject{
				{
					Object:    "embedding",
					Embedding: []float32{0.1, 0.2, 0.3},
					Index:     0,
				},
				{
					Object:    "embedding",
					Embedding: []float32{0.4, 0.5, 0.6},
					Index:     1,
				},
			},
			Model: req.Model,
			Usage: voyageai.UsageObject{
				TotalTokens: 10,
			},
		}

		respb, err := json.Marshal(&resp)
		if err != nil {
			t.Fatal(err.Error())
		}

		w.WriteHeader(201)
		w.Write(respb)
	}))
	defer s.Close()

	cl := voyageai.NewClient(&voyageai.VoyageClientOpts{
		Key:        "APIKEY",
		TimeOut:    1500,
		MaxRetries: 3,
		BaseURL:    s.URL,
	})

	embedOpts := voyageai.EmbeddingRequestOpts{
		EncodingFormat:  voyageai.Opt("test_encoding"),
		InputType:       voyageai.Opt("test input type"),
		OutputDimension: voyageai.Opt(4242),
		OutputDType:     voyageai.Opt("test dtype"),
		Truncation:      voyageai.Opt(false),
	}

	_, err := cl.Embed([]string{"input1", "input2"}, "test-model", &embedOpts)
	if err != nil {
		t.Fatal(err.Error())
	}
}

func TestRerankRequiredArgsResponse(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageai.RerankRequest
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal("Could not read request body")
		}

		err = json.Unmarshal(b, &req)
		if err != nil {
			t.Fatalf("Invalid request body")
		}

		if req.Documents == nil {
			t.Errorf("Expected non-nil value for 'Documents'")
		}

		if req.Model == "" {
			t.Errorf("Expected non-empty value for 'Model'")
		}

		if req.Query == "" {
			t.Errorf("Expected non-empty value for 'Query'")
		}

		resp := voyageai.RerankResponse{
			Object: "list",
			Data: []voyageai.RerankObject{
				{
					RelevanceScore: 0.1,
					Index:          0,
				},
				{
					RelevanceScore: 0.1,
					Index:          0,
				},
			},
			Model: req.Model,
			Usage: voyageai.UsageObject{
				TotalTokens: 10,
			},
		}

		respb, err := json.Marshal(&resp)
		if err != nil {
			t.Fatal(err.Error())
		}

		w.WriteHeader(201)
		w.Write(respb)
	}))
	defer s.Close()

	cl := voyageai.NewClient(&voyageai.VoyageClientOpts{
		Key:        "APIKEY",
		TimeOut:    1500,
		MaxRetries: 3,
		BaseURL:    s.URL,
	})

	_, err := cl.Rerank("query", []string{"input1", "input2"}, "test-model", nil)
	if err != nil {
		t.Fatal(err.Error())
	}
}

func TestRerankCustomArgsResponse(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageai.RerankRequest
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal("Could not read request body")
		}
		err = json.Unmarshal(b, &req)
		if err != nil {
			t.Fatal("Invalid request body")
		}

		if req.Documents == nil {
			t.Error("Expected non-nil value for 'Documents'")
		}

		if req.Model == "" {
			t.Errorf("Expected non-empty value for 'Model'")
		}

		if req.Query == "" {
			t.Error("Expected non-empty value for 'Query'")
		}

		if req.TopK == nil {
			t.Error("Expected non-nil value for 'TopK'")
		}

		if req.ReturnDocuments == nil {
			t.Error("Expected non-nil value for 'ReturnDocuments'")
		}

		if req.Truncation == nil {
			t.Error("Expected non-nil value for 'Truncation'")
		}

		resp := voyageai.RerankResponse{
			Object: "list",
			Data: []voyageai.RerankObject{
				{
					RelevanceScore: 0.1,
					Index:          0,
				},
				{
					RelevanceScore: 0.1,
					Index:          0,
				},
			},
			Model: req.Model,
			Usage: voyageai.UsageObject{
				TotalTokens: 10,
			},
		}

		respb, err := json.Marshal(&resp)
		if err != nil {
			t.Fatal(err.Error())
		}

		w.WriteHeader(201)
		w.Write(respb)
	}))
	defer s.Close()

	cl := voyageai.NewClient(&voyageai.VoyageClientOpts{
		Key:        "APIKEY",
		TimeOut:    1500,
		MaxRetries: 3,
		BaseURL:    s.URL,
	})

	opts := voyageai.RerankRequestOpts{
		TopK:            voyageai.Opt(3),
		ReturnDocuments: voyageai.Opt(false),
		Truncation:      voyageai.Opt(false),
	}

	_, err := cl.Rerank("query", []string{"input1", "input2"}, "test-model", &opts)
	if err != nil {
		t.Fatal(err.Error())
	}
}

func createDummyImage(width int, height int) (*bytes.Buffer, error) {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			img.Set(x, y, color.White)
		}
	}

	buf := new(bytes.Buffer)
	if err := png.Encode(buf, img); err != nil {
		return nil, err
	}
	return buf, nil
}

func validateDataURL(url string) (bool, error) {
	res := true
	strs := strings.Split(url, ",")
	header := strs[0]
	data := strs[1]
	mtype := strings.Split(strings.Split(header, ":")[1], ";")[0]
	b64 := strings.Split(header, ";")[1]

	res = res && (strings.Split(header, ":")[0] == "data")
	res = res && (b64 == "base64")
	res = res && (mtype == "image/jpeg" || mtype == "image/png" || mtype == "image/gif")

	_, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return false, err
	}
	return res, nil
}

func TestMultimodalRequiredArgsRequest(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageai.MultimodalRequest
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal("Could not read request body")
		}

		err = json.Unmarshal(b, &req)
		if err != nil {
			t.Fatalf("Invalid request body")
		}

		if req.Inputs == nil {
			t.Fatal("Expected non-nil value for 'Inputs'")
		}

		if req.Model == "" {
			t.Fatal("Expected non-nil value for 'Model'")
		}

		v, err := validateDataURL(string(req.Inputs[0].Content[0].ImageBase64))
		if err != nil {
			t.Fatal(err.Error())
		}

		if !v {
			t.Error("Invalid data url")
		}

		resp := voyageai.EmbeddingResponse{
			Object: "list",
			Data: []voyageai.EmbeddingObject{
				{
					Object:    "embedding",
					Embedding: []float32{0.1, 0.2, 0.3},
					Index:     0,
				},
				{
					Object:    "embedding",
					Embedding: []float32{0.4, 0.5, 0.6},
					Index:     1,
				},
			},
			Model: req.Model,
			Usage: voyageai.UsageObject{
				TotalTokens: 10,
			},
		}

		respb, err := json.Marshal(&resp)
		if err != nil {
			t.Fatal(err.Error())
		}

		w.WriteHeader(201)
		w.Write(respb)
	}))
	defer s.Close()

	cl := voyageai.NewClient(&voyageai.VoyageClientOpts{
		Key:        "APIKEY",
		TimeOut:    1500,
		MaxRetries: 3,
		BaseURL:    s.URL,
	})

	dummyImage1, err := createDummyImage(rand.Intn(1200), rand.Intn(630))
	if err != nil {
		t.Fatalf("Couldn't create test image: %s", err.Error())
	}

	dummyImage2, err := createDummyImage(350, 350)
	if err != nil {
		t.Fatalf("Couldn't create test image: %s", err.Error())
	}

	inputs := []voyageai.MultimodalContent{
		{
			Content: []voyageai.MultimodalInput{
				voyageai.Multimodal(voyageai.MustGetBase64(dummyImage1)),
				voyageai.Multimodal(voyageai.MustGetBase64(dummyImage2)),
			},
		},
	}

	_, err = cl.MultimodalEmbed(inputs, "test-model", nil)
	if err != nil {
		t.Fatal(err.Error())
	}
}

func TestMultimodalRequiredCustomRequest(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageai.MultimodalRequest
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal("Could not read request body")
		}

		err = json.Unmarshal(b, &req)
		if err != nil {
			t.Fatalf("Invalid request body")
		}

		if req.Inputs == nil {
			t.Fatal("Expected non-nil value for 'Inputs'")
		}

		if req.Model == "" {
			t.Fatal("Expected non-nil value for 'Model'")
		}

		if req.Truncation == nil {
			t.Fatal("Expected non-nil value for 'Truncation'")
		}

		if req.OuputEncoding == nil {
			t.Fatal("Expected non-nil value for 'OutputEncoding'")
		}

		if req.InputType == nil {
			t.Fatal("Expected non-nil value for 'InputType'")
		}

		v, err := validateDataURL(string(req.Inputs[0].Content[0].ImageBase64))
		if err != nil {
			t.Fatal(err.Error())
		}

		if !v {
			t.Error("Invalid data url")
		}

		resp := voyageai.EmbeddingResponse{
			Object: "list",
			Data: []voyageai.EmbeddingObject{
				{
					Object:    "embedding",
					Embedding: []float32{0.1, 0.2, 0.3},
					Index:     0,
				},
				{
					Object:    "embedding",
					Embedding: []float32{0.4, 0.5, 0.6},
					Index:     1,
				},
			},
			Model: req.Model,
			Usage: voyageai.UsageObject{
				TotalTokens: 10,
			},
		}

		respb, err := json.Marshal(&resp)
		if err != nil {
			t.Fatal(err.Error())
		}

		w.WriteHeader(201)
		w.Write(respb)
	}))
	defer s.Close()

	cl := voyageai.NewClient(&voyageai.VoyageClientOpts{
		Key:        "APIKEY",
		TimeOut:    1500,
		MaxRetries: 3,
		BaseURL:    s.URL,
	})

	dummyImage1, err := createDummyImage(rand.Intn(1200), rand.Intn(630))
	if err != nil {
		t.Fatalf("Couldn't create test image: %s", err.Error())
	}

	dummyImage2, err := createDummyImage(350, 350)
	if err != nil {
		t.Fatalf("Couldn't create test image: %s", err.Error())
	}

	inputs := []voyageai.MultimodalContent{
		{
			Content: []voyageai.MultimodalInput{
				voyageai.Multimodal(voyageai.MustGetBase64(dummyImage1)),
				voyageai.Multimodal(voyageai.MustGetBase64(dummyImage2)),
			},
		},
	}

	opts := voyageai.MultimodalRequestOpts{
		InputType:     voyageai.Opt("Test type"),
		Truncation:    voyageai.Opt(false),
		OuputEncoding: voyageai.Opt("base64"),
	}

	_, err = cl.MultimodalEmbed(inputs, "test-model", &opts)
	if err != nil {
		t.Fatal(err.Error())
	}
}

func TestMaxRetries(t *testing.T) {
	retries := 0
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "" {
			apiErr := voyageai.APIError{Detail: "User unauthorized"}
			b, err := json.Marshal(apiErr)
			if err != nil {
				t.Fatalf("Could not create error response")
			}
			w.WriteHeader(401)
			w.Write(b)
		}

		var req voyageai.EmbeddingRequest
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal("Could not read request body")
		}

		err = json.Unmarshal(b, &req)
		if err != nil {
			t.Fatalf("Invalid request body")
		}
		resp := voyageai.EmbeddingResponse{
			Object: "list",
			Data: []voyageai.EmbeddingObject{
				{
					Object:    "embedding",
					Embedding: []float32{0.1, 0.2, 0.3},
					Index:     0,
				},
				{
					Object:    "embedding",
					Embedding: []float32{0.4, 0.5, 0.6},
					Index:     1,
				},
			},
			Model: req.Model,
			Usage: voyageai.UsageObject{
				TotalTokens: 10,
			},
		}

		respb, err := json.Marshal(&resp)
		if err != nil {
			t.Fatal(err.Error())
		}

		// Count current request in retries
		retries++

		// Status code of 500 will result in a retry
		w.WriteHeader(500)
		w.Write(respb)
	}))
	defer s.Close()

	maxRetries := rand.Intn(10) + 1
	cl := voyageai.NewClient(&voyageai.VoyageClientOpts{
		Key:        "APIKEY",
		TimeOut:    1500,
		MaxRetries: maxRetries,
		BaseURL:    s.URL,
	})

	_, err := cl.Embed([]string{"input1", "input2"}, "test-model", nil)
	if err == nil {
		t.Fatal("Expected an error after exhausting retries, got nil")
	}

	if retries != maxRetries {
		t.Errorf("Expected retries to equal %d but got %d", maxRetries, retries)
	}
}

func TestNoRetryOnSuccess(t *testing.T) {
	requests := 0
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++

		var req voyageai.EmbeddingRequest
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal("Could not read request body")
		}

		err = json.Unmarshal(b, &req)
		if err != nil {
			t.Fatalf("Invalid request body")
		}
		resp := voyageai.EmbeddingResponse{
			Object: "list",
			Data: []voyageai.EmbeddingObject{
				{
					Object:    "embedding",
					Embedding: []float32{0.1, 0.2, 0.3},
					Index:     0,
				},
				{
					Object:    "embedding",
					Embedding: []float32{0.4, 0.5, 0.6},
					Index:     1,
				},
			},
			Model: req.Model,
			Usage: voyageai.UsageObject{
				TotalTokens: 10,
			},
		}

		respb, err := json.Marshal(&resp)
		if err != nil {
			t.Fatal(err.Error())
		}

		// Status code of 200 on the first attempt should not trigger a retry.
		w.WriteHeader(200)
		w.Write(respb)
	}))
	defer s.Close()

	maxRetries := rand.Intn(10) + 2
	cl := voyageai.NewClient(&voyageai.VoyageClientOpts{
		Key:        "APIKEY",
		TimeOut:    1500,
		MaxRetries: maxRetries,
		BaseURL:    s.URL,
	})

	_, err := cl.Embed([]string{"input1", "input2"}, "test-model", nil)
	if err != nil {
		t.Fatal(err.Error())
	}

	if requests != 1 {
		t.Errorf("Expected exactly 1 request on success but got %d", requests)
	}
}
