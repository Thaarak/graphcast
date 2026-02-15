"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Cloud, Wind, Droplets } from "lucide-react"
import { cn } from "@/lib/utils"

export interface Strategy {
  id: string
  name: string
  description: string
  icon: "cloud" | "wind" | "droplets"
  isOptimal: boolean
  kpis: {
    soilMoisture: string
    windPeak: string
    confidence: string
  }
  mitigationLevel: number
  variants?: { id: string; name: string }[]
}

const iconMap = {
  cloud: Cloud,
  wind: Wind,
  droplets: Droplets,
}

interface StrategyCardsProps {
  strategies: Strategy[]
  activeId: string
  onSelect: (id: string) => void
  sidebarTitle?: string
  selectedVariant?: string | null
  onVariantSelect?: (variant: string) => void
  checkboxMode?: boolean
  modelRun?: boolean
  onModelRun?: () => void
  checkedStrategies?: Set<string>
  onCheckChange?: (id: string, checked: boolean) => void
}

export function StrategyCards({
  strategies,
  activeId,
  onSelect,
  sidebarTitle = "Strategies",
  selectedVariant,
  onVariantSelect,
  checkboxMode = false,
  modelRun = false,
  onModelRun,
  checkedStrategies = new Set(),
  onCheckChange,
}: StrategyCardsProps) {
  if (checkboxMode) {
    return (
      <aside className="flex flex-col w-[260px] shrink-0 p-3 border-r border-border bg-background overflow-y-auto">
        <p className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground px-1 mb-2">
          {sidebarTitle}
        </p>
        <div className="flex flex-col gap-3 flex-1">
          {strategies.map((s) => (
            <label
              key={s.id}
              className="flex items-center gap-3 px-2 py-2 rounded-md hover:bg-muted/50 cursor-pointer"
            >
              <Checkbox
                id={s.id}
                checked={checkedStrategies.has(s.id)}
                onCheckedChange={(checked) => onCheckChange?.(s.id, checked === true)}
              />
              <span className="text-sm text-foreground">{s.name}</span>
            </label>
          ))}
        </div>
        <div className="mt-auto pt-4">
          <Button
            onClick={onModelRun}
            className="w-full"
            disabled={modelRun}
          >
            Model
          </Button>
        </div>
      </aside>
    )
  }

  return (
    <aside className="flex flex-col gap-2.5 w-[260px] shrink-0 p-3 border-r border-border bg-background overflow-y-auto">
      <p className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground px-1 mb-0.5">
        {sidebarTitle}
      </p>
      {strategies.map((s) => {
        const Icon = iconMap[s.icon]
        const isActive = s.id === activeId
        return (
          <div
            key={s.id}
            role="button"
            tabIndex={0}
            onClick={() => onSelect(s.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault()
                onSelect(s.id)
              }
            }}
            className="text-left w-full"
          >
            <Card
              className={cn(
                "relative transition-all duration-200 cursor-pointer",
                isActive
                  ? "border-primary/40 shadow-sm ring-1 ring-primary/10"
                  : "border-border hover:border-border/80 hover:shadow-sm"
              )}
            >
              {s.isOptimal && (
                <Badge className="absolute top-2 right-2 bg-primary/10 text-primary border-transparent text-[10px] font-medium px-1.5 py-0 leading-5">
                  Optimal
                </Badge>
              )}
              <CardHeader className="p-3 pb-1.5">
                <div className="flex items-center gap-2">
                  <div
                    className={cn(
                      "flex items-center justify-center h-7 w-7 rounded-md",
                      isActive
                        ? "bg-primary/10 text-primary"
                        : "bg-muted text-muted-foreground"
                    )}
                  >
                    <Icon className="h-3.5 w-3.5" />
                  </div>
                  <CardTitle className="text-xs font-semibold text-foreground">
                    {s.name}
                  </CardTitle>
                </div>
              </CardHeader>
              <CardContent className={cn("px-3 pb-3 pt-0", !s.description && "pb-2")}>
                {s.description && (
                  <p className="text-[11px] leading-relaxed text-muted-foreground">
                    {s.description}
                  </p>
                )}
                {isActive && s.variants && s.variants.length > 0 && onVariantSelect && (
                  <div
                    className="mt-2 pt-2 border-t border-border"
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                  >
                    <Select
                      value={selectedVariant ?? undefined}
                      onValueChange={onVariantSelect}
                    >
                      <SelectTrigger className="h-8 text-xs">
                        <SelectValue placeholder="Select scenario" />
                      </SelectTrigger>
                      <SelectContent>
                        {s.variants.map((v) => (
                          <SelectItem key={v.id} value={v.id} className="text-xs">
                            {v.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                {isActive && s.description && (
                  <div className="flex items-center gap-2 mt-2 pt-2 border-t border-border">
                    <span className="text-[10px] font-medium text-primary bg-primary/5 px-1.5 py-0.5 rounded">
                      {s.kpis.confidence} conf.
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )
      })}
    </aside>
  )
}
