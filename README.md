# 🖼️ GPT-Image MCP Server

MCP server for OpenAI's image generation API — defaults to **`gpt-image-2`** (flagship as of April 2026), with full support for `gpt-image-1`, `gpt-image-1-mini`, and the legacy DALL-E models.

Inspired by [`nanobanana-mcp`](https://github.com/waimakers/nanobanana-mcp) (Gemini), but uses OpenAI's simpler multipart upload (no separate Files API).

## ✨ Features

- 🎨 **Text-to-image** with `generate_image` (1K/2K/4K, transparent backgrounds, multiple variants)
- ✏️ **Image editing & composition** with `edit_image` (up to 16 reference images, optional mask for inpainting)
- 📐 **Flexible sizes** including 1024×1024, 1536×1024, 2K, 4K, and aspect-ratio variants
- 🎚️ **Quality tiers** (low/medium/high/auto) and output formats (png/jpeg/webp + compression)
- 🔍 **Built-in capabilities query** so the model can ask the server what it supports
- 💾 **Save to disk** automatically when `outputPath` is provided

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/waimakers/gpt-image.git
cd gpt-image
npm install
npm run build
```

### 2. Get your API key

Create one at <https://platform.openai.com/api-keys>.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
# Optional:
OPENAI_ORGANIZATION=org_...
OPENAI_PROJECT=proj_...
```

### 4. Wire into Claude Code / Cursor

Add to `~/.claude.json` (or `~/.cursor/mcp.json`) under `mcpServers`:

```json
{
  "mcpServers": {
    "gpt-image": {
      "command": "node",
      "args": ["C:\\Users\\you\\Githubs\\gpt-image\\dist\\index.js"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## 🛠️ Tools

### `generate_image`

Text-to-image. Returns base64 (or saves to disk if `outputPath` given).

| Parameter | Type | Notes |
|---|---|---|
| `prompt` | string (req) | ≤ 32,000 chars for gpt-image-* |
| `model` | enum | `gpt-image-2` (default), `gpt-image-1.5`, `gpt-image-1`, `gpt-image-1-mini`, `dall-e-3`, `dall-e-2` |
| `n` | int | 1–8 (gpt-image-2), 1–10 (gpt-image-1), 1 (dall-e-3) |
| `size` | enum | `auto`, `1024x1024`, `1536x1024`, `1024x1536`, `2048x2048`, `2560x1440`, `1440x2560`, `4096x2304`, `2304x4096`, … |
| `quality` | enum | `low`, `medium`, `high`, `auto` (gpt-image-*); `standard`, `hd` (dall-e-3) |
| `background` | enum | `transparent`, `opaque`, `auto` — transparent ⇒ png/webp |
| `output_format` | enum | `png`, `jpeg`, `webp` |
| `output_compression` | int | 0–100 (jpeg/webp only) |
| `moderation` | enum | `low`, `auto` |
| `partial_images` | int | 0–3 (streaming partials, advanced) |
| `user` | string | End-user id for abuse monitoring |
| `outputPath` | string | If set, saves PNG/JPEG/WebP to disk |

**Example:**

```json
{
  "prompt": "A serene mountain landscape at sunset, photographic",
  "model": "gpt-image-2",
  "size": "1536x1024",
  "quality": "high",
  "outputPath": "C:/Users/you/Pictures/sunset.png"
}
```

### `edit_image`

Image-to-image editing & composition. 1–16 input images, optional mask for inpainting.

| Parameter | Type | Notes |
|---|---|---|
| `prompt` | string (req) | Edit instructions |
| `inputImages` | array (req) | 1–16 images, each `{source, url|filePath|base64, mimeType?}` |
| `mask` | object | PNG with alpha; transparent pixels = areas to edit |
| `model` | enum | Same set as generate_image |
| `n`, `size`, `quality`, `background`, `output_format`, `output_compression`, `user`, `outputPath` | — | Same as generate_image |

**Image source types** (`inputImages[].source`):

- `file_path` → use `filePath: "C:/path/to/image.png"`
- `url` → use `url: "https://…/image.png"` (server downloads)
- `inline` → use `base64: "iVBORw0KG…"` (use sparingly)

**Example — combine 3 reference images:**

```json
{
  "prompt": "Combine these into a single product photo on a wooden table",
  "inputImages": [
    {"source": "file_path", "filePath": "C:/imgs/mug.png"},
    {"source": "file_path", "filePath": "C:/imgs/saucer.png"},
    {"source": "url", "url": "https://example.com/spoon.jpg"}
  ],
  "size": "1024x1024",
  "quality": "high",
  "outputPath": "C:/imgs/composed.png"
}
```

**Example — inpainting:**

```json
{
  "prompt": "Replace the masked area with a blue sky",
  "inputImages": [{"source": "file_path", "filePath": "C:/imgs/photo.png"}],
  "mask": {"source": "file_path", "filePath": "C:/imgs/photo-mask.png"},
  "outputPath": "C:/imgs/photo-fixed.png"
}
```

### `get_model_capabilities`

Returns the catalogue of models, sizes, quality tiers, output formats, and pricing notes.

## 💰 Pricing (April 2026, verify on docs)

Token-based, per 1M tokens (approximate):

| Model | Text in | Image in | Image out |
|---|---|---|---|
| `gpt-image-1` | $5 | $10 | $40 |
| `gpt-image-1-mini` | $2 | $2.50 | $8 |
| `gpt-image-2` | — | $8 | $32 |

**Approx per-image cost (1024×1024)** on gpt-image-2: low ≈ $0.006, medium ≈ $0.05, high ≈ $0.21.

Always verify at <https://platform.openai.com/docs/pricing>.

## 📊 Rate limits (gpt-image-2)

| Tier | Tokens/min | Images/min |
|---|---|---|
| 1 | 100K | 5 |
| 2 | 250K | 20 |
| 3 | 800K | 50 |
| 4 | 3M | 150 |
| 5 | 8M | 250 |

## 🔒 Limits & validation

- **Max input images per edit**: 16
- **Max file size per image**: 25 MB
- **Max prompt length**: 32,000 chars (gpt-image-*); 4,000 (dall-e-3)
- **Transparent background** requires `output_format` ∈ {`png`, `webp`}
- **Response format**: gpt-image-* always returns `b64_json` (no URLs)

## 🆘 Troubleshooting

**`OPENAI_API_KEY environment variable is required`** → Update your MCP config `env` block.

**`File not found`** → Use absolute paths for `filePath`.

**Slow first call** → Cold start; subsequent calls are faster.

**Transparent background ignored** → Force `output_format: "png"`.

## 📚 Resources

- [OpenAI Images API reference](https://platform.openai.com/docs/api-reference/images)
- [Image generation guide](https://platform.openai.com/docs/guides/image-generation)
- [Pricing](https://platform.openai.com/docs/pricing)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## 📝 License

MIT
