#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ErrorCode,
  McpError,
  Tool,
  CallToolResult,
} from "@modelcontextprotocol/sdk/types.js";
import {
  OpenAIImageClient,
  GenerateImageRequest,
  EditImageRequest,
} from "./openai-client.js";
import * as fs from "fs";
import * as path from "path";

const MODEL_ENUM = [
  "chatgpt-image-latest",
  "gpt-image-2",
  "gpt-image-2-2026-04-21",
  "gpt-image-1.5",
  "gpt-image-1",
  "gpt-image-1-mini",
  "dall-e-3",
  "dall-e-2",
];

const SIZE_ENUM = [
  "auto",
  "1024x1024",
  "1024x1536",
  "1536x1024",
  "1792x1024",
  "1024x1792",
  "2048x2048",
  "2560x1440",
  "1440x2560",
  "4096x2304",
  "2304x4096",
  "256x256",
  "512x512",
];

const QUALITY_ENUM = ["auto", "low", "medium", "high", "standard", "hd"];
const BACKGROUND_ENUM = ["transparent", "opaque", "auto"];
const OUTPUT_FORMAT_ENUM = ["png", "jpeg", "webp"];
const MODERATION_ENUM = ["low", "auto"];

const IMAGE_INPUT_SCHEMA = {
  type: "object",
  properties: {
    source: {
      type: "string",
      enum: ["url", "file_path", "inline"],
      description:
        "Where the image comes from: 'url' (download), 'file_path' (local file), 'inline' (base64).",
    },
    url: { type: "string", description: "URL to download (if source='url')" },
    filePath: {
      type: "string",
      description: "Local file path (if source='file_path')",
    },
    base64: {
      type: "string",
      description: "Base64 image data (if source='inline')",
    },
    mimeType: {
      type: "string",
      description: "MIME type, optional. Auto-detected for files/URLs.",
    },
  },
  required: ["source"],
};

const TOOLS: Tool[] = [
  {
    name: "generate_image",
    description:
      "Generate an image from a text prompt using OpenAI's image API. Default model: chatgpt-image-latest (rolling alias to OpenAI's newest image model). Response includes cost_estimate_usd. Returns base64 image; saves to disk if outputPath provided.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description:
            "Text description of the image (≤32,000 chars for gpt-image-*).",
        },
        model: {
          type: "string",
          enum: MODEL_ENUM,
          description:
            "Model to use. Default: chatgpt-image-latest (rolling alias). Pin to gpt-image-2-2026-04-21 for reproducibility.",
        },
        n: {
          type: "number",
          description:
            "Number of images to generate. 1-8 for gpt-image-2, 1-10 for gpt-image-1, 1 for dall-e-3. Default: 1.",
        },
        size: {
          type: "string",
          enum: SIZE_ENUM,
          description:
            "Output dimensions. gpt-image-2 supports 1K/2K/4K and flexible ratios. Default: auto.",
        },
        quality: {
          type: "string",
          enum: QUALITY_ENUM,
          description:
            "low | medium | high | auto for gpt-image-*; standard | hd for dall-e-3. Default: auto.",
        },
        background: {
          type: "string",
          enum: BACKGROUND_ENUM,
          description:
            "transparent | opaque | auto. Transparent requires output_format=png or webp.",
        },
        output_format: {
          type: "string",
          enum: OUTPUT_FORMAT_ENUM,
          description: "png | jpeg | webp. Default: png.",
        },
        output_compression: {
          type: "number",
          description:
            "0–100 (jpeg/webp only). Default: 100 (no compression).",
        },
        moderation: {
          type: "string",
          enum: MODERATION_ENUM,
          description: "Content moderation strictness. Default: auto.",
        },
        partial_images: {
          type: "number",
          description:
            "0–3, number of partial frames during streaming (advanced).",
        },
        user: {
          type: "string",
          description: "End-user identifier (abuse monitoring).",
        },
        outputPath: {
          type: "string",
          description:
            "Optional. Save image to this path. If omitted, base64 is returned.",
        },
      },
      required: ["prompt"],
    },
  },
  {
    name: "edit_image",
    description:
      "Edit one or more images with a prompt, optionally with a mask for inpainting. Up to 16 input images for gpt-image-2. Mask: PNG with alpha — transparent pixels = areas to edit.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: { type: "string", description: "Edit instructions." },
        inputImages: {
          type: "array",
          description: "1–16 images to edit / compose.",
          items: IMAGE_INPUT_SCHEMA,
          minItems: 1,
          maxItems: 16,
        },
        mask: {
          ...IMAGE_INPUT_SCHEMA,
          description:
            "Optional mask. PNG with alpha; transparent pixels mark areas to edit.",
        },
        model: { type: "string", enum: MODEL_ENUM },
        n: { type: "number" },
        size: { type: "string", enum: SIZE_ENUM },
        quality: { type: "string", enum: QUALITY_ENUM },
        background: { type: "string", enum: BACKGROUND_ENUM },
        output_format: { type: "string", enum: OUTPUT_FORMAT_ENUM },
        output_compression: { type: "number" },
        user: { type: "string" },
        outputPath: { type: "string" },
      },
      required: ["prompt", "inputImages"],
    },
  },
  {
    name: "get_model_capabilities",
    description:
      "Returns available models, supported sizes, quality tiers, output formats, pricing notes, and limits.",
    inputSchema: { type: "object", properties: {} },
  },
];

class GptImageServer {
  private imageClient: OpenAIImageClient;
  private server: Server;

  constructor() {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      console.error("Error: OPENAI_API_KEY environment variable is required");
      console.error(
        "Get your API key at: https://platform.openai.com/api-keys"
      );
      process.exit(1);
    }

    const organization = process.env.OPENAI_ORGANIZATION;
    const project = process.env.OPENAI_PROJECT;

    this.imageClient = new OpenAIImageClient(apiKey, organization, project);
    this.server = new Server(
      { name: "gpt-image-mcp-server", version: "0.1.0" },
      { capabilities: { tools: {} } }
    );

    this.setupHandlers();
    this.setupErrorHandling();
  }

  private setupHandlers(): void {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: TOOLS,
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) =>
      this.handleToolCall(request.params.name, request.params.arguments ?? {})
    );
  }

  private async handleToolCall(
    name: string,
    args: any
  ): Promise<CallToolResult> {
    try {
      switch (name) {
        case "generate_image":
          return await this.handleGenerate(args);
        case "edit_image":
          return await this.handleEdit(args);
        case "get_model_capabilities":
          return this.handleCapabilities();
        default:
          throw new McpError(
            ErrorCode.MethodNotFound,
            `Unknown tool: ${name}`
          );
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      process.stderr.write(`Tool execution error: ${errorMessage}\n`);
      return {
        content: [{ type: "text", text: `Error: ${errorMessage}` }],
        isError: true,
      };
    }
  }

  private async handleGenerate(args: any): Promise<CallToolResult> {
    const { outputPath, ...rest } = args;
    const request: GenerateImageRequest = rest;
    const result = await this.imageClient.generateImage(request);

    return this.formatImageResult(result, outputPath, "generated");
  }

  private async handleEdit(args: any): Promise<CallToolResult> {
    const { outputPath, ...rest } = args;
    const request: EditImageRequest = rest;
    const result = await this.imageClient.editImage(request);

    return this.formatImageResult(result, outputPath, "edited");
  }

  private formatImageResult(
    result: any,
    outputPath: string | undefined,
    verb: string
  ): CallToolResult {
    if (!result.images || result.images.length === 0) {
      throw new Error("No image data in response");
    }

    if (outputPath) {
      const savedPaths: string[] = [];
      const baseDir = path.dirname(outputPath);
      const ext = path.extname(outputPath);
      const baseName = path.basename(outputPath, ext);

      if (!fs.existsSync(baseDir)) {
        fs.mkdirSync(baseDir, { recursive: true });
      }

      result.images.forEach((img: any, i: number) => {
        const target =
          result.images.length === 1
            ? outputPath
            : path.join(baseDir, `${baseName}-${i + 1}${ext}`);
        fs.writeFileSync(target, Buffer.from(img.b64_json, "base64"));
        savedPaths.push(path.resolve(target));
      });

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Image ${verb} and saved successfully!`,
                outputPaths: savedPaths,
                count: result.images.length,
                size: result.size,
                quality: result.quality,
                output_format: result.output_format,
                background: result.background,
                revised_prompts: result.images
                  .map((i: any) => i.revised_prompt)
                  .filter(Boolean),
                cost_estimate_usd: result.cost_estimate_usd,
                usage: result.usage,
              },
              null,
              2
            ),
          },
        ],
      };
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(
            {
              success: true,
              message: `Image ${verb} successfully!`,
              count: result.images.length,
              images: result.images.map((i: any) => ({
                b64_json: i.b64_json,
                revised_prompt: i.revised_prompt,
              })),
              size: result.size,
              quality: result.quality,
              output_format: result.output_format,
              background: result.background,
              cost_estimate_usd: result.cost_estimate_usd,
              usage: result.usage,
              note:
                "Provide 'outputPath' to save to disk instead of returning base64.",
            },
            null,
            2
          ),
        },
      ],
    };
  }

  private handleCapabilities(): CallToolResult {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(
            {
              models: [
                {
                  name: "chatgpt-image-latest",
                  default: true,
                  description:
                    "Rolling alias to OpenAI's newest image model. Currently points to gpt-image-2. Use for always-latest behavior.",
                  features: [
                    "text-to-image",
                    "multi-image-edit",
                    "inpainting",
                    "transparent-background",
                    "streaming",
                  ],
                },
                {
                  name: "gpt-image-2",
                  description:
                    "Flagship image model (April 2026). 1K/2K/4K, up to 16 ref images, near-perfect text rendering.",
                  features: [
                    "text-to-image",
                    "multi-image-edit",
                    "inpainting",
                    "transparent-background",
                    "streaming",
                  ],
                  maxN: 8,
                },
                {
                  name: "gpt-image-2-2026-04-21",
                  description:
                    "Pinned snapshot of gpt-image-2 (2026-04-21). Use for reproducibility.",
                  features: [
                    "text-to-image",
                    "multi-image-edit",
                    "inpainting",
                    "transparent-background",
                  ],
                },
                {
                  name: "gpt-image-1.5",
                  description: "Mid-tier refresh of v1.",
                  features: ["text-to-image", "multi-image-edit", "inpainting"],
                },
                {
                  name: "gpt-image-1",
                  description:
                    "Original token-priced multimodal image model (April 2025).",
                  features: ["text-to-image", "multi-image-edit", "inpainting"],
                  maxN: 10,
                },
                {
                  name: "gpt-image-1-mini",
                  description: "Cheaper, faster variant. ~5–10× lower cost.",
                  features: ["text-to-image", "edit"],
                },
                {
                  name: "dall-e-3",
                  description: "Legacy. URL responses, n=1, no edit endpoint.",
                  features: ["text-to-image"],
                  maxN: 1,
                },
                {
                  name: "dall-e-2",
                  description: "Legacy. Only model supporting /variations.",
                  features: ["text-to-image", "edit", "variations"],
                },
              ],
              sizes: SIZE_ENUM,
              qualities: QUALITY_ENUM,
              backgrounds: BACKGROUND_ENUM,
              outputFormats: OUTPUT_FORMAT_ENUM,
              moderation: MODERATION_ENUM,
              limits: {
                maxReferenceImages: 16,
                maxFileSizeMb: 25,
                maxPromptChars: 32000,
              },
              pricing_per_1m_tokens_usd: {
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
              },
              approx_per_image_usd_1024x1024: {
                "gpt-image-2": { low: 0.006, medium: 0.053, high: 0.211 },
                "gpt-image-1.5": { low: 0.009, medium: 0.034, high: 0.133 },
                "gpt-image-1": { low: 0.011, medium: 0.042, high: 0.167 },
                "gpt-image-1-mini": {
                  low: 0.005,
                  medium: 0.011,
                  high: 0.036,
                },
              },
              pricing_per_image_usd: {
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
              },
              pricing_snapshot_date: "2026-04-29",
              approx_cost_per_image_1024_usd: {
                low: 0.006,
                medium: 0.05,
                high: 0.21,
              },
              notes: [
                "gpt-image-* always returns b64_json (no URLs).",
                "background='transparent' requires output_format='png' or 'webp'.",
                "Pricing verified 2026-04-29 against https://developers.openai.com/api/docs/pricing.",
                "OpenAI's API does NOT return USD cost — the MCP computes cost_estimate_usd from token counts × rates.",
                "Re-verify before billing: https://platform.openai.com/docs/pricing",
              ],
            },
            null,
            2
          ),
        },
      ],
    };
  }

  private setupErrorHandling(): void {
    this.server.onerror = (error) => {
      process.stderr.write(`[MCP Error] ${error}\n`);
    };

    process.on("SIGINT", async () => {
      await this.server.close();
      process.exit(0);
    });

    process.on("uncaughtException", (error) => {
      process.stderr.write(`[Uncaught Exception] ${error.message}\n`);
      process.exit(1);
    });

    process.on("unhandledRejection", (reason) => {
      process.stderr.write(`[Unhandled Rejection] ${reason}\n`);
    });
  }

  async start(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    process.stderr.write("🖼️  GPT-Image MCP server is running\n");
  }
}

async function main() {
  const server = new GptImageServer();
  await server.start();
}

main().catch((error) => {
  process.stderr.write(`Fatal server error: ${error.message}\n`);
  process.exit(1);
});
