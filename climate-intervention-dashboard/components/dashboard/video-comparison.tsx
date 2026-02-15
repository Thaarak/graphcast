"use client"

import { useRef, useEffect, useState } from "react"
import { Play, Pause } from "lucide-react"
import { Button } from "@/components/ui/button"

interface VideoComparisonProps {
  controlSrc: string
  seededSrc: string
  strategyName: string
  playbackRate?: number
}

export function VideoComparison({ controlSrc, seededSrc, strategyName, playbackRate = 1 }: VideoComparisonProps) {
  const controlVideoRef = useRef<HTMLVideoElement>(null)
  const seededVideoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(true)

  // Set playback rate
  useEffect(() => {
    const control = controlVideoRef.current
    const seeded = seededVideoRef.current
    if (control) control.playbackRate = playbackRate
    if (seeded) seeded.playbackRate = playbackRate
  }, [playbackRate])

  const togglePlayPause = () => {
    const control = controlVideoRef.current
    const seeded = seededVideoRef.current
    if (!control || !seeded) return

    if (isPlaying) {
      control.pause()
      seeded.pause()
    } else {
      control.play()
      seeded.play()
    }
    setIsPlaying(!isPlaying)
  }

  // Sync video playback
  useEffect(() => {
    const control = controlVideoRef.current
    const seeded = seededVideoRef.current
    if (!control || !seeded) return

    const syncVideos = () => {
      if (Math.abs(control.currentTime - seeded.currentTime) > 0.1) {
        seeded.currentTime = control.currentTime
      }
    }

    control.addEventListener("timeupdate", syncVideos)
    control.addEventListener("play", () => seeded.play())
    control.addEventListener("pause", () => seeded.pause())
    control.addEventListener("seeked", () => {
      seeded.currentTime = control.currentTime
    })

    return () => {
      control.removeEventListener("timeupdate", syncVideos)
    }
  }, [])

  return (
    <div className="flex flex-1 bg-black overflow-hidden">
      {/* Control (Before) */}
      <div className="relative flex-1 flex flex-col">
        <div className="absolute top-3 left-3 z-10">
          <div className="bg-background/90 backdrop-blur-sm border border-border rounded-md px-2.5 py-1.5 text-[11px] font-semibold text-red-500">
            CONTROL FORECAST
          </div>
        </div>
        <video
          ref={controlVideoRef}
          src={controlSrc}
          className="flex-1 w-full h-full object-contain"
          autoPlay
          loop
          muted
          playsInline
        />
      </div>

      {/* Divider */}
      <div className="w-px bg-border" />

      {/* Seeded (After) */}
      <div className="relative flex-1 flex flex-col">
        <div className="absolute top-3 right-3 z-10">
          <div className="bg-background/90 backdrop-blur-sm border border-border rounded-md px-2.5 py-1.5 text-[11px] font-semibold text-blue-500">
            SEEDED FORECAST
          </div>
        </div>
        <video
          ref={seededVideoRef}
          src={seededSrc}
          className="flex-1 w-full h-full object-contain"
          autoPlay
          loop
          muted
          playsInline
        />
      </div>

      {/* Strategy label overlay */}
      <div className="absolute bottom-3 left-3 z-10">
        <div className="bg-background/90 backdrop-blur-sm border border-border rounded-md px-2.5 py-1.5 text-[11px] font-medium text-muted-foreground">
          Active: <span className="text-foreground">{strategyName}</span>
        </div>
      </div>

      {/* Play/Pause button */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10">
        <Button
          variant="secondary"
          size="icon"
          onClick={togglePlayPause}
          className="h-10 w-10 rounded-full bg-background/90 backdrop-blur-sm border border-border hover:bg-background"
        >
          {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
        </Button>
      </div>

    </div>
  )
}
