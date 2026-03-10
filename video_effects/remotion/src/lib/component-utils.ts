/**
 * Utilities for diagram, timeline, and code block components.
 */

/**
 * Generate an SVG path `d` string for a curved connector between two points.
 * @param x1 - Start x
 * @param y1 - Start y
 * @param x2 - End x
 * @param y2 - End y
 * @param curve - Curvature factor (0 = straight line, default 0.5)
 */
export function drawConnector(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  curve: number = 0.5,
): string {
  if (curve === 0) {
    return `M ${x1} ${y1} L ${x2} ${y2}`;
  }
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  // Offset the control point perpendicular to the line
  const cx = midX - dy * curve * 0.5;
  const cy = midY + dx * curve * 0.5;
  return `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
}

/**
 * Compute evenly distributed positions along a range.
 * @param count - Number of points
 * @param start - Range start
 * @param end - Range end
 * @returns Array of evenly-spaced positions
 */
export function distributeEvenly(
  count: number,
  start: number,
  end: number,
): number[] {
  if (count <= 0) return [];
  if (count === 1) return [(start + end) / 2];
  const step = (end - start) / (count - 1);
  return Array.from({ length: count }, (_, i) => start + i * step);
}

type TokenType = "keyword" | "string" | "comment" | "number" | "plain";

interface Token {
  text: string;
  type: TokenType;
}

const KEYWORDS: Record<string, Set<string>> = {
  javascript: new Set([
    "const", "let", "var", "function", "return", "if", "else", "for", "while",
    "class", "import", "export", "from", "default", "async", "await", "new",
    "this", "true", "false", "null", "undefined", "typeof", "instanceof",
    "switch", "case", "break", "continue", "try", "catch", "finally", "throw",
    "of", "in", "yield",
  ]),
  typescript: new Set([
    "const", "let", "var", "function", "return", "if", "else", "for", "while",
    "class", "import", "export", "from", "default", "async", "await", "new",
    "this", "true", "false", "null", "undefined", "typeof", "instanceof",
    "switch", "case", "break", "continue", "try", "catch", "finally", "throw",
    "of", "in", "yield", "type", "interface", "enum", "as", "implements",
    "extends", "readonly", "private", "public", "protected", "abstract",
  ]),
  python: new Set([
    "def", "class", "return", "if", "elif", "else", "for", "while", "import",
    "from", "as", "with", "try", "except", "finally", "raise", "pass", "break",
    "continue", "and", "or", "not", "in", "is", "lambda", "yield", "async",
    "await", "True", "False", "None", "self",
  ]),
};

/**
 * Simple regex-based tokenizer for syntax highlighting.
 * Supports JavaScript, TypeScript, and Python.
 * @param code - Source code string
 * @param language - Programming language
 * @returns Array of tokens with text and type
 */
export function tokenize(code: string, language: string): Token[] {
  const lang = language.toLowerCase();
  const keywords = KEYWORDS[lang] || KEYWORDS["javascript"] || new Set();
  const tokens: Token[] = [];

  // Combined regex: comments, strings, numbers, words, whitespace/other
  const pattern =
    lang === "python"
      ? /(#[^\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_]\w*\b|[^\s\w]+|\s+)/g
      : /(\/\/[^\n]*|\/\*[\s\S]*?\*\/|`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b\d+(?:\.\d+)?\b|\b[a-zA-Z_$]\w*\b|[^\s\w]+|\s+)/g;

  let match: RegExpExecArray | null;
  while ((match = pattern.exec(code)) !== null) {
    const text = match[0];
    let type: TokenType = "plain";

    if (
      text.startsWith("//") ||
      text.startsWith("/*") ||
      text.startsWith("#")
    ) {
      type = "comment";
    } else if (
      text.startsWith('"') ||
      text.startsWith("'") ||
      text.startsWith("`") ||
      text.startsWith('"""') ||
      text.startsWith("'''")
    ) {
      type = "string";
    } else if (/^\d/.test(text)) {
      type = "number";
    } else if (keywords.has(text)) {
      type = "keyword";
    }

    tokens.push({ text, type });
  }

  return tokens;
}
