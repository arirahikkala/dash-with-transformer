import { Scene, Band } from "./scene";

/** Stable hue derived from a character code. */
function bandColor(charCode: number): string {
  const hue = (charCode * 137.508) % 360; // golden-angle spread
  return `hsl(${hue}, 45%, 35%)`;
}

function bandColorAlt(charCode: number): string {
  const hue = (charCode * 137.508) % 360;
  return `hsl(${hue}, 45%, 30%)`;
}

/** Render a Scene onto a 2D canvas. */
export function renderScene(
  ctx: CanvasRenderingContext2D,
  scene: Scene,
  width: number,
  height: number,
): void {
  ctx.clearRect(0, 0, width, height);

  // -- Draw bands --
  for (let i = 0; i < scene.bands.length; i++) {
    const band = scene.bands[i];
    const py0 = band.y0 * height;
    const py1 = band.y1 * height;
    const bandHeight = py1 - py0;

    // Band fill
    ctx.fillStyle = i % 2 === 0 ? bandColor(band.charCode) : bandColorAlt(band.charCode);
    ctx.fillRect(0, py0, width, bandHeight);

    // Band border (subtle line at top)
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, py0);
    ctx.lineTo(width, py0);
    ctx.stroke();

    // Label — only draw if the band is tall enough to fit text
    drawBandLabel(ctx, band, py0, bandHeight, width);
  }

  // -- Crosshairs --
  const cx = scene.crosshairs.x * width;
  const cy = scene.crosshairs.y * height;

  ctx.strokeStyle = "rgba(255, 60, 60, 0.8)";
  ctx.lineWidth = 1.5;

  // Vertical line
  ctx.beginPath();
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, height);
  ctx.stroke();

  // Horizontal line
  ctx.beginPath();
  ctx.moveTo(0, cy);
  ctx.lineTo(width, cy);
  ctx.stroke();
}

function drawBandLabel(
  ctx: CanvasRenderingContext2D,
  band: Band,
  py0: number,
  bandHeight: number,
  width: number,
): void {
  // Pick a font size that fits, clamped to reasonable range
  const fontSize = Math.min(Math.max(bandHeight * 0.7, 10), 28);
  if (bandHeight < 10) return; // too small to label

  ctx.font = `${fontSize}px monospace`;
  ctx.fillStyle = "#e0e0e0";
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.fillText(band.label, width / 2, py0 + bandHeight / 2);
}
