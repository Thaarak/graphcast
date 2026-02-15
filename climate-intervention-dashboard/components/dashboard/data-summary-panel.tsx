"use client"

import { CheckCircle, TrendingUp, Droplets, Wind, TrendingDown } from "lucide-react"

interface InterventionData {
  precip_increase_pct: number
  cells_resolved: number
  cells_improved: number
  cells_worsened: number
  isBest: boolean
}

interface ControlData {
  drought_cells: number
  severity_score: number
}

interface KatrinaData {
  event: string
  forecastPeriod: string
  windSpeedReduction: {
    initial: number
    peak: number
    average: number
    peakHour: number
  }
}

interface CycloneExperimentData {
  event: string
  forecastPeriod: string
  avgWindReduction: number // m/s average across forecast
}

interface DataSummaryPanelProps {
  region?: string
  forecastDate?: string
  control?: ControlData
  intervention?: InterventionData
  interventionName?: string
  allInterventions?: Record<string, InterventionData>
  showAllComparison?: boolean
  katrinaData?: KatrinaData
  katrinaVariant?: "control" | "seeded"
  cycloneExperimentData?: CycloneExperimentData
}

const interventionDisplayNames: Record<string, string> = {
  electric_ionization: "Electric Ionization",
  glaciogenic_static: "Glaciogenic",
  hygroscopic_enhancement: "Hygroscopic",
  laser_induced_condensation: "Laser Induced",
}

export function DataSummaryPanel({
  region,
  forecastDate,
  control,
  intervention,
  interventionName,
  allInterventions,
  showAllComparison = false,
  katrinaData,
  katrinaVariant,
  cycloneExperimentData,
}: DataSummaryPanelProps) {
  // Cyclone experiment data display
  if (cycloneExperimentData) {
    const { event, forecastPeriod, avgWindReduction } = cycloneExperimentData

    return (
      <div className="bg-slate-900 border-b border-slate-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            {/* Event & Forecast Period */}
            <div className="flex items-center gap-4 text-xs text-slate-400">
              <span>Event: <span className="text-white font-medium">{event}</span></span>
              <span>Forecast: <span className="text-white font-medium">{forecastPeriod}</span></span>
            </div>

            {/* Divider */}
            <div className="h-8 w-px bg-slate-700" />

            {/* Average Wind Reduction */}
            <div className="flex items-center gap-2">
              <Wind className="h-4 w-4 text-cyan-400" />
              <span className="text-xs text-slate-400">Avg Wind Reduction:</span>
              <span className="text-sm font-semibold text-green-400 bg-green-400/10 px-2 py-0.5 rounded">
                -{avgWindReduction} m/s
              </span>
            </div>

          </div>

          {/* Label */}
          <div className="text-xs font-medium text-blue-400 bg-blue-400/10 px-2 py-1 rounded">
            Experimental
          </div>
        </div>
      </div>
    )
  }

  // Hurricane Katrina data display
  if (katrinaData) {
    const { event, forecastPeriod, windSpeedReduction } = katrinaData
    const isSeeded = katrinaVariant === "seeded"

    return (
      <div className="bg-slate-900 border-b border-slate-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            {/* Event & Forecast Period */}
            <div className="flex items-center gap-4 text-xs text-slate-400">
              <span>Event: <span className="text-white font-medium">{event}</span></span>
              <span>Forecast: <span className="text-white font-medium">{forecastPeriod}</span></span>
            </div>

            {isSeeded && (
              <>
                {/* Divider */}
                <div className="h-8 w-px bg-slate-700" />

                {/* Wind Speed Reduction */}
                <div className="flex items-center gap-2">
                  <Wind className="h-4 w-4 text-cyan-400" />
                  <span className="text-xs text-slate-400">Wind Reduction:</span>
                  <span className="text-sm font-semibold text-green-400 bg-green-400/10 px-2 py-0.5 rounded">
                    {windSpeedReduction.average}% avg
                  </span>
                </div>

                {/* Divider */}
                <div className="h-8 w-px bg-slate-700" />

                {/* Peak Reduction */}
                <div className="flex items-center gap-2">
                  <TrendingDown className="h-4 w-4 text-green-400" />
                  <span className="text-xs text-slate-400">Peak:</span>
                  <span className="text-sm font-semibold text-green-400">{windSpeedReduction.peak}%</span>
                  <span className="text-xs text-slate-500">at hour {windSpeedReduction.peakHour}</span>
                </div>

                {/* Divider */}
                <div className="h-8 w-px bg-slate-700" />

                {/* Effect Indicator */}
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-emerald-400" />
                  <span className="text-xs text-emerald-400 font-medium">Effect strengthens over time</span>
                </div>
              </>
            )}

            {!isSeeded && (
              <>
                {/* Divider */}
                <div className="h-8 w-px bg-slate-700" />

                {/* Control baseline info */}
                <div className="flex items-center gap-2">
                  <Wind className="h-4 w-4 text-slate-400" />
                  <span className="text-xs text-slate-400">Baseline forecast (no intervention)</span>
                </div>
              </>
            )}
          </div>

          {/* Variant Label */}
          <div className="flex items-center gap-2">
            {isSeeded && (
              <span className="text-xs bg-green-600 text-white px-2 py-1 rounded font-medium">
                SEEDED
              </span>
            )}
            <div className={`text-xs font-medium px-2 py-1 rounded ${
              isSeeded
                ? "text-green-400 bg-green-400/10"
                : "text-slate-400 bg-slate-700/50"
            }`}>
              {isSeeded ? "Cloud Seeding Applied" : "Control"}
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (showAllComparison && allInterventions && control) {
    // Show comparison of all interventions sorted by precipitation increase
    const sortedInterventions = Object.entries(allInterventions).sort(
      (a, b) => b[1].precip_increase_pct - a[1].precip_increase_pct
    )

    return (
      <div className="bg-slate-900 border-b border-slate-700 px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <span>Region: <span className="text-white font-medium">{region}</span></span>
            <span>Forecast: <span className="text-white font-medium">{forecastDate}</span></span>
            <span>Baseline Drought Cells: <span className="text-white font-medium">{control.drought_cells.toLocaleString()}</span></span>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-3">
          {sortedInterventions.map(([key, data], index) => (
            <div
              key={key}
              className={`rounded px-3 py-2 border ${
                index === 0
                  ? "bg-green-900/30 border-green-600"
                  : "bg-slate-800/50 border-slate-700"
              }`}
            >
              <div className="flex items-center gap-1.5 mb-1">
                <span className={`text-xs font-medium ${index === 0 ? "text-green-300" : "text-slate-300"}`}>
                  {interventionDisplayNames[key] || key}
                </span>
                {index === 0 && (
                  <span className="text-[10px] bg-green-600 text-white px-1.5 py-0.5 rounded font-medium">
                    BEST
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5">
                <Droplets className={`h-3 w-3 ${index === 0 ? "text-green-400" : "text-blue-400"}`} />
                <span className={`text-sm font-semibold ${index === 0 ? "text-green-400" : "text-blue-400"}`}>
                  +{data.precip_increase_pct}%
                </span>
                <span className="text-xs text-slate-500">precip</span>
              </div>
              <div className="flex items-center gap-3 mt-1 text-xs text-slate-400">
                <span><span className="text-green-400 font-medium">{data.cells_resolved}</span> resolved</span>
                <span><span className="text-emerald-400 font-medium">{data.cells_improved}</span> improved</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (!intervention || !interventionName || !control) {
    return null
  }

  return (
    <div className="bg-slate-900 border-b border-slate-700 px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6">
          {/* Region & Date */}
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <span>Region: <span className="text-white font-medium">{region}</span></span>
            <span>Forecast: <span className="text-white font-medium">{forecastDate}</span></span>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-slate-700" />

          {/* Precipitation Increase */}
          <div className="flex items-center gap-2">
            <Droplets className="h-4 w-4 text-blue-400" />
            <span className="text-xs text-slate-400">Precipitation:</span>
            <span className="text-sm font-semibold text-green-400 bg-green-400/10 px-2 py-0.5 rounded">
              +{intervention.precip_increase_pct}%
            </span>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-slate-700" />

          {/* Cells Resolved */}
          <div className="flex items-center gap-2">
            <CheckCircle className="h-4 w-4 text-green-400" />
            <span className="text-xs text-slate-400">Cells Resolved:</span>
            <span className="text-sm font-semibold text-green-400">{intervention.cells_resolved}</span>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-slate-700" />

          {/* Cells Improved */}
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-emerald-400" />
            <span className="text-xs text-slate-400">Cells Improved:</span>
            <span className="text-sm font-semibold text-emerald-400">{intervention.cells_improved}</span>
          </div>

          {/* Cells Worsened (only show if > 0) */}
          {intervention.cells_worsened > 0 && (
            <>
              <div className="h-8 w-px bg-slate-700" />
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">Worsened:</span>
                <span className="text-sm font-semibold text-red-400">{intervention.cells_worsened}</span>
              </div>
            </>
          )}
        </div>

        {/* Intervention Name */}
        <div className="flex items-center gap-2">
          {intervention.isBest && (
            <span className="text-xs bg-green-600 text-white px-2 py-1 rounded font-medium">
              BEST
            </span>
          )}
          <div className="text-xs font-medium text-blue-400 bg-blue-400/10 px-2 py-1 rounded">
            {interventionName}
          </div>
        </div>
      </div>
    </div>
  )
}
