import OpenAI, { toFile } from "openai";
import axios from "axios";
import * as fs from "fs";
import * as path from "path";

export type ImageModel =
  | "chatgpt-image-latest"
  | "gpt-image-2"
  | "gpt-image-2-2026-04-21"
  | "gpt-image-1.5"
  | "gpt-image-1"
  | "gpt-image-1-mini"
  | "dall-e-3"
  | "dall-e-2";

export type ImageSize =
  | "auto"
  | "1024x1024"
  | "1024x1536"
  | "1536x1024"
  | "1792x1024"
  | "1024x1792"
  | "2048x2048"
  | "2560x1440"
  | "1440x2560"
  | "4096x2304"
  | "2304x4096"
  | "256x256"
  | "512x512";

export type ImageQuality =
  | "auto"
  | "low"
  | "medium"
  | "high"
  | "standard"
  | "hd";

export type ImageBackground = "transparent" | "opaque" | "auto";
export type ImageOutputFormat = "png" | "jpeg" | "webp";
export type ImageModeration = "low" | "auto";

export interface ImageInput {
  source: "url" | "file_path" | "inline";
  url?: string;
  filePath?: string;
  base64?: string;
  mimeType?: string;
}

export interface GenerateImageRequest {
  prompt: string;
  model?: ImageModel;
  n?: number;
  size?: ImageSize;
  quality?: ImageQuality;
  background?: ImageBackground;
  output_format?: ImageOutputFormat;
  output_compression?: number;
  moderation?: ImageModeration;
  user?: string;
  partial_images?: number;
}

export interface EditImageRequest {
  prompt: string;
  inputImages: ImageInput[];
  mask?: ImageInput;
  model?: ImageModel;
  n?: number;
  size?: ImageSize;
  quality?: ImageQuality;
  background?: ImageBackground;
  output_format?: ImageOutputFormat;
  output_compression?: number;
  user?: string;
}

export interface ImageResult {
  images: Array<{ b64_json: string; revised_prompt?: string }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
    input_tokens_details?: any;
    output_tokens_details?: any;
  };
  cost_estimate_usd?: {
    total: number;
    breakdown: any;
    pricing_source_date: string;
    note: string;
  } | null;
  size?: string;
  quality?: string;
  output_format?: string;
  background?: string;
}

// Default to the rolling alias so the MCP automatically rides any future
// OpenAI image model upgrade without a code change.
const DEFAULT_MODEL: ImageModel = "chatgpt-image-latest";

// Pricing snapshot, USD per 1M tokens.
// Verified 2026-04-29 against:
//   - https://developers.openai.com/api/docs/pricing            (primary, OpenAI dev docs)
//   - https://community.openai.com/t/gpt-image-1-collected-pricing-information-and-why-responses-is-undocumented/1275254  (gpt-image-1, legacy)
//   - https://techcrunch.com/2025/04/23/openai-makes-its-upgraded-image-generator-available-to-developers/  (gpt-image-1 launch)
//   - https://www.imagine.art/blogs/gpt-image-1-5-pricing       (gpt-image-1.5 cross-check)
//   - https://www.eesel.ai/blog/gpt-image-1-mini-pricing        (gpt-image-1-mini cross-check)
const PRICING_DATE = "2026-04-29";

interface TokenRates {
  text_in?: number;
  text_out?: number;
  image_in?: number;
  image_out: number;
  cached_text_in?: number;
  cached_image_in?: number;
}

const PRICING_PER_1M_TOKENS: Record<string, TokenRates> = {
  // gpt-image-2 (released 2026-04-21). image_out is $30, NOT $32 — corrected
  // from earlier third-party sources.
  "chatgpt-image-latest": {
    text_in: 5,
    image_in: 8,
    image_out: 30,
    cached_image_in: 2,
  },
  "gpt-image-2": {
    text_in: 5,
    image_in: 8,
    image_out: 30,
    cached_image_in: 2,
  },
  "gpt-image-2-2026-04-21": {
    text_in: 5,
    image_in: 8,
    image_out: 30,
    cached_image_in: 2,
  },
  // gpt-image-1.5 also bills text_out (it's a reasoning model).
  // Corrected: image_in is $8 (not $10), image_out is $32 (not $36).
  "gpt-image-1.5": {
    text_in: 5,
    text_out: 10,
    image_in: 8,
    image_out: 32,
    cached_text_in: 1.25,
    cached_image_in: 2,
  },
  "gpt-image-1": {
    text_in: 5,
    image_in: 10,
    image_out: 40,
    cached_text_in: 1.25,
    cached_image_in: 2.5,
  },
  "gpt-image-1-mini": {
    text_in: 2,
    image_in: 2.5,
    image_out: 8,
    cached_image_in: 0.25,
  },
};

// DALL-E uses flat per-image pricing instead of tokens.
const PRICING_PER_IMAGE: Record<string, Record<string, number>> = {
  "dall-e-3": {
    "standard:1024x1024": 0.04,
    "hd:1024x1024": 0.08,
    "standard:1024x1792": 0.08,
    "hd:1024x1792": 0.12,
    "standard:1792x1024": 0.08,
    "hd:1792x1024": 0.12,
  },
  "dall-e-2": {
    "1024x1024": 0.02,
    "512x512": 0.018,
    "256x256": 0.016,
  },
};

function round4(n: number): number {
  return Number(n.toFixed(4));
}

function estimateCost(
  model: string,
  usage: any | undefined,
  n: number,
  size: string | undefined,
  quality: string | undefined
): ImageResult["cost_estimate_usd"] {
  const tokenRates = PRICING_PER_1M_TOKENS[model];
  if (tokenRates && usage) {
    // Total reported tokens, before splitting cached / fresh.
    const textInTotal = usage.input_tokens_details?.text_tokens ?? 0;
    const imageInTotal = usage.input_tokens_details?.image_tokens ?? 0;
    const imageOut =
      usage.output_tokens_details?.image_tokens ?? usage.output_tokens ?? 0;
    const textOut = usage.output_tokens_details?.text_tokens ?? 0;

    // OpenAI returns the cached portion alongside the totals when caching applies.
    // The fresh portion is total - cached, billed at the higher rate.
    const cachedTextIn =
      usage.input_tokens_details?.cached_tokens?.text_tokens ??
      usage.input_tokens_details?.cached_text_tokens ??
      0;
    const cachedImageIn =
      usage.input_tokens_details?.cached_tokens?.image_tokens ??
      usage.input_tokens_details?.cached_image_tokens ??
      0;

    const freshTextIn = Math.max(0, textInTotal - cachedTextIn);
    const freshImageIn = Math.max(0, imageInTotal - cachedImageIn);

    const textInUsd =
      (freshTextIn / 1_000_000) * (tokenRates.text_in ?? 0) +
      (cachedTextIn / 1_000_000) * (tokenRates.cached_text_in ?? 0);
    const imageInUsd =
      (freshImageIn / 1_000_000) * (tokenRates.image_in ?? 0) +
      (cachedImageIn / 1_000_000) * (tokenRates.cached_image_in ?? 0);
    const imageOutUsd = (imageOut / 1_000_000) * tokenRates.image_out;
    const textOutUsd = (textOut / 1_000_000) * (tokenRates.text_out ?? 0);

    const total = textInUsd + imageInUsd + imageOutUsd + textOutUsd;

    return {
      total: round4(total),
      breakdown: {
        text_input_tokens: textInTotal,
        image_input_tokens: imageInTotal,
        cached_text_input_tokens: cachedTextIn,
        cached_image_input_tokens: cachedImageIn,
        image_output_tokens: imageOut,
        text_output_tokens: textOut,
        text_input_usd: round4(textInUsd),
        image_input_usd: round4(imageInUsd),
        image_output_usd: round4(imageOutUsd),
        text_output_usd: round4(textOutUsd),
        rates_per_1m_tokens: tokenRates,
      },
      pricing_source_date: PRICING_DATE,
      note:
        "Verified 2026-04-29 against developers.openai.com/api/docs/pricing. Re-verify before using for billing.",
    };
  }

  const flatRates = PRICING_PER_IMAGE[model];
  if (flatRates && size) {
    const key =
      model === "dall-e-3" && quality ? `${quality}:${size}` : size;
    const perImage = flatRates[key];
    if (perImage !== undefined) {
      return {
        total: round4(perImage * n),
        breakdown: { per_image_usd: perImage, n, key },
        pricing_source_date: PRICING_DATE,
        note: `Flat per-image rate for ${model}.`,
      };
    }
  }

  return null;
}

export class OpenAIImageClient {
  private client: OpenAI;

  constructor(apiKey: string, organization?: string, project?: string) {
    this.client = new OpenAI({
      apiKey,
      organization: organization || undefined,
      project: project || undefined,
    });
  }

  async generateImage(request: GenerateImageRequest): Promise<ImageResult> {
    const params: any = {
      model: request.model ?? DEFAULT_MODEL,
      prompt: request.prompt,
    };
    if (request.n !== undefined) params.n = request.n;
    if (request.size) params.size = request.size;
    if (request.quality) params.quality = request.quality;
    if (request.background) params.background = request.background;
    if (request.output_format) params.output_format = request.output_format;
    if (request.output_compression !== undefined)
      params.output_compression = request.output_compression;
    if (request.moderation) params.moderation = request.moderation;
    if (request.user) params.user = request.user;
    if (request.partial_images !== undefined)
      params.partial_images = request.partial_images;

    if (
      params.background === "transparent" &&
      params.output_format === "jpeg"
    ) {
      throw new Error(
        "Transparent background requires output_format='png' or 'webp', not 'jpeg'"
      );
    }

    process.stderr.write(`🎨 Generating image with ${params.model}...\n`);
    const res: any = await this.client.images.generate(params);
    process.stderr.write(`✅ Image generated (${res.data?.length ?? 0} variant(s))\n`);

    const n = params.n ?? 1;
    const cost = estimateCost(
      params.model,
      res.usage,
      n,
      res.size ?? params.size,
      res.quality ?? params.quality
    );

    return {
      images: (res.data ?? []).map((d: any) => ({
        b64_json: d.b64_json,
        revised_prompt: d.revised_prompt,
      })),
      usage: res.usage,
      cost_estimate_usd: cost,
      size: res.size,
      quality: res.quality,
      output_format: res.output_format,
      background: res.background,
    };
  }

  async editImage(request: EditImageRequest): Promise<ImageResult> {
    if (!request.inputImages || request.inputImages.length === 0) {
      throw new Error("At least one inputImage is required for edit");
    }
    if (request.inputImages.length > 16) {
      throw new Error("Maximum 16 input images supported");
    }

    const imageFiles = await Promise.all(
      request.inputImages.map((img, i) =>
        this.resolveToFile(img, `image-${i}.png`)
      )
    );

    const params: any = {
      model: request.model ?? DEFAULT_MODEL,
      prompt: request.prompt,
      image: imageFiles.length === 1 ? imageFiles[0] : imageFiles,
    };

    if (request.mask) {
      params.mask = await this.resolveToFile(request.mask, "mask.png");
    }
    if (request.n !== undefined) params.n = request.n;
    if (request.size) params.size = request.size;
    if (request.quality) params.quality = request.quality;
    if (request.background) params.background = request.background;
    if (request.output_format) params.output_format = request.output_format;
    if (request.output_compression !== undefined)
      params.output_compression = request.output_compression;
    if (request.user) params.user = request.user;

    process.stderr.write(
      `✏️  Editing with ${params.model} (${imageFiles.length} image(s)${
        request.mask ? " + mask" : ""
      })...\n`
    );
    const res: any = await this.client.images.edit(params);
    process.stderr.write(`✅ Edit complete (${res.data?.length ?? 0} variant(s))\n`);

    const n = params.n ?? 1;
    const cost = estimateCost(
      params.model,
      res.usage,
      n,
      res.size ?? params.size,
      res.quality ?? params.quality
    );

    return {
      images: (res.data ?? []).map((d: any) => ({
        b64_json: d.b64_json,
        revised_prompt: d.revised_prompt,
      })),
      usage: res.usage,
      cost_estimate_usd: cost,
      size: res.size,
      quality: res.quality,
      output_format: res.output_format,
      background: res.background,
    };
  }

  private async resolveToFile(input: ImageInput, defaultName: string) {
    if (input.source === "file_path" && input.filePath) {
      if (!fs.existsSync(input.filePath)) {
        throw new Error(`File not found: ${input.filePath}`);
      }
      const buffer = fs.readFileSync(input.filePath);
      const name = path.basename(input.filePath);
      return await toFile(buffer, name, {
        type: input.mimeType || this.getMimeType(name),
      });
    }
    if (input.source === "url" && input.url) {
      process.stderr.write(`Downloading from URL: ${input.url}\n`);
      const response = await axios.get(input.url, {
        responseType: "arraybuffer",
      });
      const buffer = Buffer.from(response.data);
      const mime =
        (response.headers["content-type"] as string | undefined) ||
        input.mimeType ||
        "image/png";
      return await toFile(buffer, defaultName, { type: mime });
    }
    if (input.source === "inline" && input.base64) {
      const buffer = Buffer.from(input.base64, "base64");
      return await toFile(buffer, defaultName, {
        type: input.mimeType || "image/png",
      });
    }
    throw new Error(
      "Invalid image input — must provide one of: url, filePath, base64"
    );
  }

  private getMimeType(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();
    const map: Record<string, string> = {
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".webp": "image/webp",
      ".gif": "image/gif",
    };
    return map[ext] || "application/octet-stream";
  }
}
