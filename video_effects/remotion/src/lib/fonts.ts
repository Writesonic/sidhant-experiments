import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadBebasNeue } from "@remotion/google-fonts/BebasNeue";
import { loadFont as loadDMSans } from "@remotion/google-fonts/DMSans";
import { loadFont as loadOswald } from "@remotion/google-fonts/Oswald";
import { loadFont as loadSourceSans3 } from "@remotion/google-fonts/SourceSans3";
import { loadFont as loadPoppins } from "@remotion/google-fonts/Poppins";

type FontLoader = (
  style: "normal" | "italic",
  options: { weights: string[]; subsets: string[] },
) => { fontFamily: string };

const FONT_LOADERS: Record<string, FontLoader> = {
  Inter: loadInter as unknown as FontLoader,
  BebasNeue: loadBebasNeue as unknown as FontLoader,
  DMSans: loadDMSans as unknown as FontLoader,
  Oswald: loadOswald as unknown as FontLoader,
  SourceSans3: loadSourceSans3 as unknown as FontLoader,
  Poppins: loadPoppins as unknown as FontLoader,
};

export function loadStyleFont(
  fontImport: string,
  weights: string[],
): string {
  const loader = FONT_LOADERS[fontImport];
  if (!loader) return "sans-serif";
  const { fontFamily } = loader("normal", { weights, subsets: ["latin"] });
  return fontFamily;
}
