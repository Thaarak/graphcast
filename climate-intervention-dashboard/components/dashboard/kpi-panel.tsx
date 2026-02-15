import { Droplets, Wind, ShieldCheck } from "lucide-react"

interface KpiPanelProps {
  soilMoisture: string
  windPeak: string
  confidence: string
}

export function KpiPanel({ soilMoisture, windPeak, confidence }: KpiPanelProps) {
  const kpis = [
    {
      label: "Soil Moisture",
      value: soilMoisture,
      icon: Droplets,
      color: "text-primary",
    },
    {
      label: "Wind Peak",
      value: windPeak,
      icon: Wind,
      color: "text-foreground",
    },
    {
      label: "Success Confidence",
      value: confidence,
      icon: ShieldCheck,
      color: "text-primary",
    },
  ]

  return (
    <div className="flex items-center gap-4">
      {kpis.map((kpi) => (
        <div
          key={kpi.label}
          className="flex items-center gap-2 bg-background border border-border rounded-md px-3 py-1.5"
        >
          <kpi.icon className={`h-3.5 w-3.5 ${kpi.color}`} />
          <div className="flex flex-col">
            <span className="text-[9px] uppercase tracking-wider text-muted-foreground leading-none">
              {kpi.label}
            </span>
            <span className="text-xs font-semibold text-foreground leading-tight mt-0.5">
              {kpi.value}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}
