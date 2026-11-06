package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/MoeForge/moego/ojson"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
)

// One server plays both the upstream and this proxy's own endpoint used by the vision fallback.
func TestOpenAICompatExecutorDescribesImagesForTextOnlyModel(t *testing.T) {
	tests := []struct {
		name            string
		inputModalities []string
		fallbackModel   string
		payload         string
		wantText        string
		wantImage       bool
	}{
		{
			name:            "user image described",
			inputModalities: []string{"text"},
			fallbackModel:   "vision-model",
			payload:         `{"model":"claude-client","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"qué dice?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"VVNFUg=="}}]}]}`,
			wantText:        "BOOTLOADER LOCKED",
		},
		{
			name:            "tool_result image described",
			inputModalities: []string{"text"},
			fallbackModel:   "vision-model",
			payload:         `{"model":"claude-client","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"Read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"VE9PTA=="}}]}]}]}`,
			wantText:        "BOOTLOADER LOCKED",
		},
		{
			name:            "user image without fallback gets placeholder",
			inputModalities: []string{"text"},
			payload:         `{"model":"claude-client","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"qué dice?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"Tk9GQg=="}}]}]}`,
			wantText:        "[imagen omitida: este modelo no acepta imágenes]",
		},
		{
			name:          "vision model keeps the image",
			fallbackModel: "vision-model",
			payload:       `{"model":"claude-client","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"qué dice?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"VklTSQ=="}}]}]}`,
			wantImage:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var mu sync.Mutex
			var upstreamBody []byte
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				if ojson.GetBytes(body, "model").String() == "vision-model" {
					if r.Header.Get("Authorization") != "Bearer proxy-key" {
						w.WriteHeader(http.StatusUnauthorized)
						return
					}
					_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"BOOTLOADER LOCKED"},"finish_reason":"stop"}]}`))
					return
				}
				mu.Lock()
				upstreamBody = body
				mu.Unlock()
				_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
			}))
			defer server.Close()
			serverURL, _ := url.Parse(server.URL)
			port, _ := strconv.Atoi(serverURL.Port())

			cfg := &config.Config{
				Port: port,
				OpenAICompatibility: []config.OpenAICompatibility{{
					Name:                "compat",
					VisionFallbackModel: tt.fallbackModel,
					Models: []config.OpenAICompatibilityModel{{
						Name:            "mapped-model",
						Alias:           "claude-client",
						InputModalities: tt.inputModalities,
					}},
				}},
			}
			cfg.APIKeys = []string{"proxy-key"}
			exec := NewOpenAICompatExecutor("openai-compatibility", cfg)
			auth := &cliproxyauth.Auth{
				Provider: "openai-compatibility",
				Attributes: map[string]string{
					"base_url": server.URL + "/v1", "api_key": "test",
					"compat_name": "compat", "provider_key": "compat",
				},
			}
			req := cliproxyexecutor.Request{Model: "mapped-model", Payload: []byte(tt.payload)}
			opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude, ResponseFormat: sdktranslator.FormatOpenAI}
			if _, err := exec.Execute(context.Background(), auth, req, opts); err != nil {
				t.Fatalf("Execute: %v", err)
			}

			mu.Lock()
			got := string(upstreamBody)
			mu.Unlock()
			if hasImage := strings.Contains(got, "image_url"); hasImage != tt.wantImage {
				t.Fatalf("image_url present = %v, want %v:\n%s", hasImage, tt.wantImage, got)
			}
			if tt.wantText != "" && !strings.Contains(got, tt.wantText) {
				t.Fatalf("upstream body lacks %q:\n%s", tt.wantText, got)
			}
		})
	}
}
