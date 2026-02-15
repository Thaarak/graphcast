"use client"

import { useCallback, useRef, useState, useEffect } from "react"
import { GripVertical } from "lucide-react"

interface MapViewProps {
  mitigationLevel: number
  strategyName: string
}

function drawMap(
  canvas: HTMLCanvasElement,
  type: "baseline" | "mitigated",
  mitigationLevel: number
) {
  const ctx = canvas.getContext("2d")
  if (!ctx) return

  const dpr = window.devicePixelRatio || 1
  const rect = canvas.getBoundingClientRect()
  if (rect.width === 0 || rect.height === 0) return
  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr
  ctx.scale(dpr, dpr)
  const w = rect.width
  const h = rect.height

  // Ocean base
  ctx.fillStyle = type === "baseline" ? "#F1F5F9" : "#F8FAFC"
  ctx.fillRect(0, 0, w, h)

  // Lat / lon grid
  ctx.strokeStyle = "#E2E8F0"
  ctx.lineWidth = 0.5
  const gridSize = 50
  for (let x = 0; x < w; x += gridSize) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, h)
    ctx.stroke()
  }
  for (let y = 0; y < h; y += gridSize) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(w, y)
    ctx.stroke()
  }

  // Coordinate labels
  ctx.font = "400 9px system-ui"
  ctx.fillStyle = "#CBD5E1"
  const lats = ["60N", "40N", "20N", "EQ", "20S", "40S"]
  const lons = ["120W", "90W", "60W", "30W", "0", "30E", "60E"]
  lats.forEach((label, i) => {
    const y = (h / (lats.length + 1)) * (i + 1)
    ctx.fillText(label, 6, y + 3)
  })
  lons.forEach((label, i) => {
    const x = (w / (lons.length + 1)) * (i + 1)
    ctx.fillText(label, x - 8, h - 6)
  })

  // Stylized landmasses
  const drawLand = (
    cx: number, cy: number,
    rw: number, rh: number,
    rot: number,
    points: number = 8
  ) => {
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(rot)
    ctx.beginPath()
    for (let i = 0; i <= points; i++) {
      const angle = (i / points) * Math.PI * 2
      const jitter = 1 + Math.sin(angle * 3 + cx) * 0.15
      const px = Math.cos(angle) * rw * jitter
      const py = Math.sin(angle) * rh * jitter
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.closePath()
    ctx.fillStyle = type === "baseline" ? "#E2E8F0" : "#E0F2FE"
    ctx.fill()
    ctx.strokeStyle = type === "baseline" ? "#CBD5E1" : "#BAE6FD"
    ctx.lineWidth = 1
    ctx.stroke()
    ctx.restore()
  }

  // North America
  drawLand(w * 0.18, h * 0.28, w * 0.1, h * 0.16, -0.15, 12)
  // South America
  drawLand(w * 0.22, h * 0.6, w * 0.05, h * 0.14, 0.1, 10)
  // Europe
  drawLand(w * 0.48, h * 0.22, w * 0.06, h * 0.08, 0.2, 10)
  // Africa
  drawLand(w * 0.5, h * 0.48, w * 0.07, h * 0.16, 0.05, 12)
  // Asia
  drawLand(w * 0.65, h * 0.25, w * 0.14, h * 0.14, -0.1, 14)
  // Australia
  drawLand(w * 0.78, h * 0.62, w * 0.06, h * 0.06, 0.3, 8)
  // Greenland
  drawLand(w * 0.32, h * 0.12, w * 0.03, h * 0.04, 0.1, 8)

  if (type === "baseline") {
    // Storm systems with spirals
    const drawStorm = (cx: number, cy: number, r: number, intensity: number) => {
      // Outer halos
      for (let i = 4; i >= 0; i--) {
        const alpha = 0.02 + intensity * 0.025 * (4 - i)
        ctx.beginPath()
        ctx.arc(cx, cy, r + i * r * 0.5, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(239, 68, 68, ${alpha})`
        ctx.fill()
      }
      // Core
      ctx.beginPath()
      ctx.arc(cx, cy, r * 0.35, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(239, 68, 68, ${0.12 + intensity * 0.1})`
      ctx.fill()
      ctx.strokeStyle = `rgba(239, 68, 68, ${0.25 + intensity * 0.15})`
      ctx.lineWidth = 1.5
      ctx.stroke()
      // Spiral arm
      ctx.beginPath()
      ctx.strokeStyle = `rgba(239, 68, 68, ${0.1 + intensity * 0.06})`
      ctx.lineWidth = 1
      for (let t = 0; t < Math.PI * 3; t += 0.1) {
        const sr = r * 0.15 + t * r * 0.12
        const sx = cx + Math.cos(t) * sr
        const sy = cy + Math.sin(t) * sr
        if (t === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
    }

    drawStorm(w * 0.28, h * 0.42, 32, 0.9)
    drawStorm(w * 0.58, h * 0.36, 26, 0.65)
    drawStorm(w * 0.73, h * 0.52, 22, 0.5)

    // Wind flow lines
    ctx.strokeStyle = "rgba(100, 116, 139, 0.2)"
    ctx.lineWidth = 1
    const flows = [
      { x1: w * 0.08, y1: h * 0.48, x2: w * 0.22, y2: h * 0.42 },
      { x1: w * 0.35, y1: h * 0.52, x2: w * 0.52, y2: h * 0.38 },
      { x1: w * 0.62, y1: h * 0.58, x2: w * 0.72, y2: h * 0.5 },
      { x1: w * 0.42, y1: h * 0.68, x2: w * 0.56, y2: h * 0.62 },
    ]
    for (const { x1, y1, x2, y2 } of flows) {
      const mx = (x1 + x2) / 2
      const my = (y1 + y2) / 2 - 8
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.quadraticCurveTo(mx, my, x2, y2)
      ctx.stroke()
      const angle = Math.atan2(y2 - my, x2 - mx)
      ctx.beginPath()
      ctx.moveTo(x2, y2)
      ctx.lineTo(x2 - 5 * Math.cos(angle - 0.4), y2 - 5 * Math.sin(angle - 0.4))
      ctx.moveTo(x2, y2)
      ctx.lineTo(x2 - 5 * Math.cos(angle + 0.4), y2 - 5 * Math.sin(angle + 0.4))
      ctx.stroke()
    }

    // Label
    ctx.font = "600 11px system-ui"
    ctx.fillStyle = "#64748B"
    ctx.fillText("BASELINE FORECAST", 16, 24)
  } else {
    const factor = mitigationLevel / 100

    // Reduced / neutralised storms
    const drawMitigated = (cx: number, cy: number, r: number, reduction: number) => {
      const eff = reduction * factor
      const newR = r * (1 - eff * 0.55)
      const alpha = Math.max(0.02, 0.06 * (1 - eff * 0.6))
      ctx.beginPath()
      ctx.arc(cx, cy, newR, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`
      ctx.fill()
      ctx.strokeStyle = `rgba(59, 130, 246, 0.12)`
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.stroke()
      ctx.setLineDash([])
    }

    drawMitigated(w * 0.28, h * 0.42, 32, 0.9)
    drawMitigated(w * 0.58, h * 0.36, 26, 0.65)
    drawMitigated(w * 0.73, h * 0.52, 22, 0.5)

    // Shield perimeters
    const drawShield = (cx: number, cy: number, r: number) => {
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.strokeStyle = `rgba(59, 130, 246, ${0.1 + factor * 0.1})`
      ctx.lineWidth = 1.5
      ctx.setLineDash([6, 4])
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = `rgba(59, 130, 246, ${0.02 + factor * 0.02})`
      ctx.fill()

      // Cross-hatch inside shield
      ctx.save()
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.clip()
      ctx.strokeStyle = `rgba(59, 130, 246, ${0.04})`
      ctx.lineWidth = 0.5
      for (let i = -r; i < r; i += 12) {
        ctx.beginPath()
        ctx.moveTo(cx + i, cy - r)
        ctx.lineTo(cx + i + r, cy + r)
        ctx.stroke()
      }
      ctx.restore()
    }

    drawShield(w * 0.28, h * 0.42, 52)
    drawShield(w * 0.58, h * 0.36, 44)
    drawShield(w * 0.73, h * 0.52, 38)

    // Label
    ctx.font = "600 11px system-ui"
    ctx.fillStyle = "#3B82F6"
    ctx.fillText("MITIGATED FORECAST", 16, 24)
  }
}

function useCanvasMap(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  type: "baseline" | "mitigated",
  mitigationLevel: number
) {
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Initial draw after a brief layout tick
    const raf = requestAnimationFrame(() => {
      drawMap(canvas, type, mitigationLevel)
    })

    // Redraw on resize
    const observer = new ResizeObserver(() => {
      drawMap(canvas, type, mitigationLevel)
    })
    observer.observe(canvas)

    return () => {
      cancelAnimationFrame(raf)
      observer.disconnect()
    }
  }, [canvasRef, type, mitigationLevel])
}

export function MapView({ mitigationLevel, strategyName }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const baselineRef = useRef<HTMLCanvasElement>(null)
  const mitigatedRef = useRef<HTMLCanvasElement>(null)
  const [sliderPos, setSliderPos] = useState(50)
  const isDragging = useRef(false)

  useCanvasMap(baselineRef, "baseline", mitigationLevel)
  useCanvasMap(mitigatedRef, "mitigated", mitigationLevel)

  const handleMove = useCallback(
    (clientX: number) => {
      if (!isDragging.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const x = ((clientX - rect.left) / rect.width) * 100
      setSliderPos(Math.max(5, Math.min(95, x)))
    },
    []
  )

  const handlePointerDown = useCallback(() => {
    isDragging.current = true
  }, [])

  useEffect(() => {
    const handlePointerMove = (e: PointerEvent) => handleMove(e.clientX)
    const handlePointerUp = () => {
      isDragging.current = false
    }
    window.addEventListener("pointermove", handlePointerMove)
    window.addEventListener("pointerup", handlePointerUp)
    return () => {
      window.removeEventListener("pointermove", handlePointerMove)
      window.removeEventListener("pointerup", handlePointerUp)
    }
  }, [handleMove])

  return (
    <div
      ref={containerRef}
      className="relative flex-1 bg-muted overflow-hidden select-none"
    >
      {/* Baseline canvas (full width, underneath) */}
      <canvas
        ref={baselineRef}
        className="absolute inset-0 w-full h-full"
      />

      {/* Mitigated canvas (clipped to right side) */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }}
      >
        <canvas
          ref={mitigatedRef}
          className="absolute inset-0 w-full h-full"
        />
      </div>

      {/* Slider handle */}
      <div
        className="absolute top-0 bottom-0 z-10 flex items-center"
        style={{ left: `${sliderPos}%`, transform: "translateX(-50%)" }}
      >
        <div className="w-px h-full bg-border" />
        <div
          onPointerDown={handlePointerDown}
          className="absolute top-1/2 -translate-y-1/2 flex items-center justify-center h-10 w-6 rounded-md bg-background border border-border shadow-sm cursor-col-resize hover:bg-muted transition-colors"
          role="slider"
          aria-label="Comparison slider"
          aria-valuenow={Math.round(sliderPos)}
          tabIndex={0}
        >
          <GripVertical className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
      </div>

      {/* Strategy label overlay */}
      <div className="absolute bottom-3 left-3 z-10">
        <div className="bg-background/90 backdrop-blur-sm border border-border rounded-md px-2.5 py-1.5 text-[11px] font-medium text-muted-foreground">
          Active: <span className="text-foreground">{strategyName}</span>
        </div>
      </div>
    </div>
  )
}
