"use client"

import { cn } from "@/lib/utils"
import { useState } from "react"

const days = [
  { label: "D1", date: "Feb 14" },
  { label: "D2", date: "Feb 15" },
  { label: "D3", date: "Feb 16" },
  { label: "D4", date: "Feb 17" },
  { label: "D5", date: "Feb 18" },
  { label: "D6", date: "Feb 19" },
  { label: "D7", date: "Feb 20" },
  { label: "D8", date: "Feb 21" },
  { label: "D9", date: "Feb 22" },
  { label: "D10", date: "Feb 23" },
]

interface TimelineProps {
  mitigationLevel: number
}

export function Timeline({ mitigationLevel }: TimelineProps) {
  const [activeDay, setActiveDay] = useState(0)

  // Simulate intensity values per day
  const getIntensity = (index: number, mitigated: boolean) => {
    const base = [0.3, 0.5, 0.7, 0.9, 1.0, 0.85, 0.7, 0.5, 0.35, 0.2]
    const val = base[index] ?? 0.5
    if (mitigated) {
      return val * (1 - mitigationLevel / 140)
    }
    return val
  }

  return (
    <div className="flex items-end gap-1 h-full px-1">
      {days.map((day, i) => {
        const baseH = getIntensity(i, false) * 28
        const mitigatedH = getIntensity(i, true) * 28
        const isActive = i === activeDay

        return (
          <button
            key={day.label}
            onClick={() => setActiveDay(i)}
            className={cn(
              "flex flex-col items-center gap-0.5 flex-1 group cursor-pointer rounded-sm py-1 transition-colors",
              isActive ? "bg-accent" : "hover:bg-muted"
            )}
          >
            <div className="flex items-end gap-px h-7">
              <div
                className="w-1.5 rounded-sm bg-destructive/25 transition-all"
                style={{ height: `${baseH}px` }}
              />
              <div
                className="w-1.5 rounded-sm bg-primary/40 transition-all"
                style={{ height: `${mitigatedH}px` }}
              />
            </div>
            <span
              className={cn(
                "text-[9px] font-medium",
                isActive ? "text-foreground" : "text-muted-foreground"
              )}
            >
              {day.label}
            </span>
          </button>
        )
      })}
    </div>
  )
}
