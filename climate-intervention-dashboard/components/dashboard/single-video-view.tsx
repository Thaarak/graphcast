"use client"

import { useRef, useState, useEffect } from "react"
import { Play, Pause } from "lucide-react"
import { Button } from "@/components/ui/button"

interface SingleVideoViewProps {
  src: string
  strategyName: string
  modelName: string
  playbackRate?: number
}

export function SingleVideoView({ src, strategyName, modelName, playbackRate = 1 }: SingleVideoViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(true)

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackRate
    }
  }, [playbackRate])

  const togglePlayPause = () => {
    const video = videoRef.current
    if (!video) return

    if (isPlaying) {
      video.pause()
    } else {
      video.play()
    }
    setIsPlaying(!isPlaying)
  }

  return (
    <div className="flex flex-1 bg-black overflow-hidden">
      <div className="relative flex-1 flex flex-col">
        <video
          ref={videoRef}
          src={src}
          className="flex-1 w-full h-full object-contain"
          autoPlay
          loop
          muted
          playsInline
        />

        {/* Title label */}
        <div className="absolute top-3 left-3 z-10">
          <div className="bg-background/90 backdrop-blur-sm border border-border rounded-md px-2.5 py-1.5 text-[11px] font-semibold text-blue-500">
            COMPARISON VIEW
          </div>
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
    </div>
  )
}
