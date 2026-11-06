package helps

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/MoeForge/moego/ojson"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	log "github.com/sirupsen/logrus"
)

const (
	// TextOnlyImageOmittedText replaces an image that could not be described for a text-only model.
	TextOnlyImageOmittedText     = "[imagen omitida: este modelo no acepta imágenes]"
	textOnlyImageDescribedHeader = "[imagen convertida a texto porque este modelo no acepta imágenes]\n"
	// claudeToolResultImageTextPlaceholder replaces the translator placeholder once the relayed
	// images have been turned into text, so the tool message stays truthful.
	claudeToolResultImageTextPlaceholder = "[Tool returned image content; it follows as text in the next user message.]"

	imageDescribeTimeout    = 30 * time.Second
	imageDescribeFailureTTL = 10 * time.Minute
	imageDescribeCacheMax   = 512
	imageDescribePrompt     = "Describí esta imagen con fidelidad para alguien que no puede verla. " +
		"Primero transcribí literalmente todo el texto visible, respetando líneas, números, rutas y símbolos. " +
		"Después describí el resto: qué tipo de pantalla o foto es, los elementos importantes y su estado. " +
		"No inventes nada que no se vea."
)

// ImageDescriber turns an image URL (data URL or http URL) into a text description.
type ImageDescriber func(ctx context.Context, imageURL string) (string, error)

type imageDescriptionEntry struct {
	text     string
	failedAt time.Time
}

// imageDescriptionCache keeps descriptions by image hash: clients resend the whole history
// every turn, so each image must be described only once.
type imageDescriptionCache struct {
	mu    sync.Mutex
	items map[string]imageDescriptionEntry
	order []string
	now   func() time.Time
}

var imageDescriptions = newImageDescriptionCache()

func newImageDescriptionCache() *imageDescriptionCache {
	return &imageDescriptionCache{items: make(map[string]imageDescriptionEntry), now: time.Now}
}

func (c *imageDescriptionCache) get(key string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.items[key]
	if !ok {
		return "", false
	}
	if entry.text == "" {
		if c.now().Sub(entry.failedAt) < imageDescribeFailureTTL {
			return TextOnlyImageOmittedText, true
		}
		return "", false
	}
	return entry.text, true
}

func (c *imageDescriptionCache) put(key string, entry imageDescriptionEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.items[key]; !exists {
		c.order = append(c.order, key)
		if len(c.order) > imageDescribeCacheMax {
			delete(c.items, c.order[0])
			c.order = c.order[1:]
		}
	}
	c.items[key] = entry
}

func (c *imageDescriptionCache) describe(ctx context.Context, describe ImageDescriber, imageURL string) string {
	sum := sha256.Sum256([]byte(imageURL))
	key := hex.EncodeToString(sum[:])
	if text, ok := c.get(key); ok {
		return text
	}
	describeCtx, cancel := context.WithTimeout(ctx, imageDescribeTimeout)
	defer cancel()
	text, err := describe(describeCtx, imageURL)
	text = strings.TrimSpace(text)
	if err != nil || text == "" {
		log.Warnf("text-only image fallback: could not describe image: %v", err)
		c.put(key, imageDescriptionEntry{failedAt: c.now()})
		return TextOnlyImageOmittedText
	}
	text = textOnlyImageDescribedHeader + text
	c.put(key, imageDescriptionEntry{text: text})
	return text
}

// ReplaceOpenAIImagesWithTextDoc replaces every image part of an OpenAI chat payload with text,
// for upstream models that reject images. With a describer each image becomes its description;
// without one, or when describing fails, it becomes TextOnlyImageOmittedText.
func ReplaceOpenAIImagesWithTextDoc(ctx context.Context, doc *ojson.Doc, describe ImageDescriber) bool {
	return replaceOpenAIImagesWithText(ctx, doc, describe, imageDescriptions)
}

func replaceOpenAIImagesWithText(ctx context.Context, doc *ojson.Doc, describe ImageDescriber, cache *imageDescriptionCache) bool {
	if doc == nil || doc.Len() == 0 {
		return false
	}
	messages := doc.Get("messages")
	if !messages.IsArray() {
		return false
	}

	type imagePart struct {
		path string
		url  string
	}
	var parts []imagePart
	for i, msg := range messages.Array() {
		content := msg.Get("content")
		if !content.IsArray() {
			continue
		}
		for j, part := range content.Array() {
			if isOpenAIImageToolResultPart(part) {
				parts = append(parts, imagePart{path: fmt.Sprintf("messages.%d.content.%d", i, j), url: openAIImagePartURL(part)})
			}
		}
	}
	if len(parts) == 0 {
		return false
	}

	// Describe each distinct image once, in parallel.
	described := make(map[string]string)
	if describe != nil {
		var mu sync.Mutex
		var wg sync.WaitGroup
		// Dedupe in a set of its own: described is written by the goroutines.
		queued := make(map[string]bool)
		for _, part := range parts {
			if part.url == "" || queued[part.url] {
				continue
			}
			queued[part.url] = true
			wg.Add(1)
			go func(url string) {
				defer wg.Done()
				text := cache.describe(ctx, describe, url)
				mu.Lock()
				described[url] = text
				mu.Unlock()
			}(part.url)
		}
		wg.Wait()
	}
	texts := make([]string, len(parts))
	for idx, part := range parts {
		texts[idx] = TextOnlyImageOmittedText
		if text, ok := described[part.url]; ok {
			texts[idx] = text
		}
	}

	modified := false
	for idx, part := range parts {
		textPart := ojson.NewDocString(`{"type":"text"}`)
		if errSet := textPart.Set("text", texts[idx]); errSet != nil {
			continue
		}
		if errSet := doc.SetRawBytes(part.path, textPart.Bytes()); errSet == nil {
			modified = true
		}
	}
	if modified {
		for i, msg := range doc.Get("messages").Array() {
			content := msg.Get("content")
			if msg.Get("role").String() == "tool" && content.Type == ojson.String && content.String() == claudeToolResultImagePlaceholder {
				_ = doc.Set(fmt.Sprintf("messages.%d.content", i), claudeToolResultImageTextPlaceholder)
			}
		}
	}
	return modified
}

func openAIImagePartURL(part ojson.Result) string {
	for _, path := range []string{"image_url.url", "image_url", "input_image.image_url", "input_image.url", "url"} {
		if value := part.Get(path); value.Type == ojson.String && value.String() != "" {
			return value.String()
		}
	}
	return ""
}

// NewLoopbackImageDescriber describes images by calling model through this same proxy, so any
// vision-capable model the proxy routes can serve as the fallback. It returns nil when model is
// empty or the proxy has no client API key to call itself with.
func NewLoopbackImageDescriber(cfg *config.Config, model string) ImageDescriber {
	model = strings.TrimSpace(model)
	if cfg == nil || model == "" || len(cfg.APIKeys) == 0 || cfg.Port <= 0 {
		return nil
	}
	scheme := "http"
	client := &http.Client{}
	if cfg.TLS.Enable {
		scheme = "https"
		// Loopback call to ourselves; the certificate is issued for the public name.
		client.Transport = &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}} // #nosec G402
	}
	endpoint := fmt.Sprintf("%s://127.0.0.1:%d/v1/chat/completions", scheme, cfg.Port)
	apiKey := cfg.APIKeys[0]
	return func(ctx context.Context, imageURL string) (string, error) {
		payload := ojson.NewDocString(`{"max_tokens":1500,"stream":false,"messages":[{"role":"user","content":[{"type":"text"},{"type":"image_url","image_url":{}}]}]}`)
		for path, value := range map[string]string{
			"model":                              model,
			"messages.0.content.0.text":          imageDescribePrompt,
			"messages.0.content.1.image_url.url": imageURL,
		} {
			if errSet := payload.Set(path, value); errSet != nil {
				return "", errSet
			}
		}
		req, errReq := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload.Bytes()))
		if errReq != nil {
			return "", errReq
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)
		resp, errDo := client.Do(req)
		if errDo != nil {
			return "", errDo
		}
		defer func() {
			if errClose := resp.Body.Close(); errClose != nil {
				log.Errorf("text-only image fallback: close response body error: %v", errClose)
			}
		}()
		data, errRead := io.ReadAll(resp.Body)
		if errRead != nil {
			return "", errRead
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return "", fmt.Errorf("vision fallback %s: status %d", model, resp.StatusCode)
		}
		return ojson.GetBytes(data, "choices.0.message.content").String(), nil
	}
}
