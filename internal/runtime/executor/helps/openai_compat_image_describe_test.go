package helps

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MoeForge/moego/ojson"
)

func countingDescriber(calls *int32, text string, err error) ImageDescriber {
	return func(_ context.Context, _ string) (string, error) {
		atomic.AddInt32(calls, 1)
		return text, err
	}
}

func TestReplaceOpenAIImagesWithText_UserImage(t *testing.T) {
	doc := ojson.NewDocString(`{"messages":[{"role":"user","content":[{"type":"text","text":"mirá"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`)
	var calls int32
	if !replaceOpenAIImagesWithText(context.Background(), doc, countingDescriber(&calls, "BOOTLOADER LOCKED", nil), newImageDescriptionCache()) {
		t.Fatal("expected modification")
	}
	out := doc.String()
	if strings.Contains(out, "image_url") {
		t.Fatalf("image part survived: %s", out)
	}
	part := doc.Get("messages.0.content.1")
	if part.Get("type").String() != "text" || !strings.Contains(part.Get("text").String(), "BOOTLOADER LOCKED") {
		t.Fatalf("unexpected part: %s", part.Raw)
	}
	if doc.Get("messages.0.content.0.text").String() != "mirá" {
		t.Fatalf("text part changed: %s", out)
	}
}

func TestReplaceOpenAIImagesWithText_ToolResultRelay(t *testing.T) {
	// Shape produced by the Claude->OpenAI translator for a tool_result carrying an image.
	doc := ojson.NewDocString(`{"messages":[` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"Read","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":"` + claudeToolResultImagePlaceholder + `"},` +
		`{"role":"user","content":[{"type":"text","text":"` + claudeToolResultImageRelayNotice + `"},{"type":"image_url","image_url":{"url":"data:image/png;base64,BBBB"}}]}]}`)
	var calls int32
	replaceOpenAIImagesWithText(context.Background(), doc, countingDescriber(&calls, "pantalla de fastboot", nil), newImageDescriptionCache())
	NormalizeOpenAIToolResultsTextOnlyDoc(doc)
	out := doc.String()
	if strings.Contains(out, "image_url") {
		t.Fatalf("image part survived: %s", out)
	}
	if !strings.Contains(out, "pantalla de fastboot") {
		t.Fatalf("description missing: %s", out)
	}
	if got := doc.Get("messages.1.content").String(); got != claudeToolResultImageTextPlaceholder {
		t.Fatalf("tool content = %q", got)
	}
	if strings.Contains(out, openAIToolResultImageOmittedText) {
		t.Fatalf("described image reported as omitted: %s", out)
	}
}

func TestReplaceOpenAIImagesWithText_CachesByImage(t *testing.T) {
	cache := newImageDescriptionCache()
	var calls int32
	describe := countingDescriber(&calls, "desc", nil)
	payload := `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,CCCC"}}]},{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,CCCC"}}]}]}`
	for turn := 0; turn < 3; turn++ {
		replaceOpenAIImagesWithText(context.Background(), ojson.NewDocString(payload), describe, cache)
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("describer called %d times, want 1", got)
	}
}

func TestReplaceOpenAIImagesWithText_FallbackPlaceholder(t *testing.T) {
	payload := `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,DDDD"}}]}]}`

	cache := newImageDescriptionCache()
	var calls int32
	doc := ojson.NewDocString(payload)
	replaceOpenAIImagesWithText(context.Background(), doc, countingDescriber(&calls, "", errors.New("boom")), cache)
	if got := doc.Get("messages.0.content.0.text").String(); got != TextOnlyImageOmittedText {
		t.Fatalf("text = %q, want placeholder", got)
	}
	// A failure is remembered for a while so every turn does not wait for the timeout again.
	replaceOpenAIImagesWithText(context.Background(), ojson.NewDocString(payload), countingDescriber(&calls, "", errors.New("boom")), cache)
	if calls != 1 {
		t.Fatalf("failed image retried immediately: %d calls", calls)
	}
	cache.now = func() time.Time { return time.Now().Add(imageDescribeFailureTTL + time.Minute) }
	replaceOpenAIImagesWithText(context.Background(), ojson.NewDocString(payload), countingDescriber(&calls, "", errors.New("boom")), cache)
	if calls != 2 {
		t.Fatalf("failed image not retried after TTL: %d calls", calls)
	}

	doc = ojson.NewDocString(payload)
	replaceOpenAIImagesWithText(context.Background(), doc, nil, newImageDescriptionCache())
	if got := doc.Get("messages.0.content.0.text").String(); got != TextOnlyImageOmittedText {
		t.Fatalf("nil describer text = %q, want placeholder", got)
	}
}

func TestReplaceOpenAIImagesWithText_NoImagesUntouched(t *testing.T) {
	payload := `{"messages":[{"role":"user","content":[{"type":"text","text":"hola"}]}]}`
	doc := ojson.NewDocString(payload)
	if replaceOpenAIImagesWithText(context.Background(), doc, nil, newImageDescriptionCache()) {
		t.Fatal("payload without images reported as modified")
	}
	if doc.String() != payload {
		t.Fatalf("payload changed: %s", doc.String())
	}
}

func TestReplaceOpenAIImagesWithText_ManyImagesConcurrently(t *testing.T) {
	var parts []string
	for i := 0; i < 300; i++ {
		parts = append(parts, `{"type":"image_url","image_url":{"url":"data:image/png;base64,`+strconv.Itoa(i)+`"}}`)
	}
	doc := ojson.NewDocString(`{"messages":[{"role":"user","content":[` + strings.Join(parts, ",") + `]}]}`)
	var calls int32
	replaceOpenAIImagesWithText(context.Background(), doc, countingDescriber(&calls, "captura", nil), newImageDescriptionCache())
	if strings.Contains(doc.String(), "image_url") {
		t.Fatalf("image part survived: %s", doc.String())
	}
}
