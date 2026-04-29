import OpenAI, { toFile } from "openai";
import axios from "axios";
import * as fs from "fs";
import * as path from "path";

export type ImageModel =
  | "gpt-image-2"
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
  };
  size?: string;
  quality?: string;
  output_format?: string;
  background?: string;
}

const DEFAULT_MODEL: ImageModel = "gpt-image-2";

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

    return {
      images: (res.data ?? []).map((d: any) => ({
        b64_json: d.b64_json,
        revised_prompt: d.revised_prompt,
      })),
      usage: res.usage,
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

    return {
      images: (res.data ?? []).map((d: any) => ({
        b64_json: d.b64_json,
        revised_prompt: d.revised_prompt,
      })),
      usage: res.usage,
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
