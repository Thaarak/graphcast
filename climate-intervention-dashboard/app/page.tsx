"use client"

import { useState, useCallback } from "react"
import { DashboardHeader, type ModelType } from "@/components/dashboard/header"
import { StrategyCards, type Strategy } from "@/components/dashboard/strategy-cards"
import { VideoComparison } from "@/components/dashboard/video-comparison"
import { SingleVideoView } from "@/components/dashboard/single-video-view"
import { DataSummaryPanel } from "@/components/dashboard/data-summary-panel"

// Hurricane Katrina data for wind speed reduction analysis
const katrinaData = {
  event: "Hurricane Katrina",
  forecastPeriod: "10 Days",
  windSpeedReduction: {
    initial: -1.25,
    peak: -2.0,
    average: -1.5,
    peakHour: 175,
  },
}

// Cyclone experiment data (experimental variant)
const cycloneExperimentData = {
  event: "Cyclone Simulation",
  forecastPeriod: "10 Days",
  avgWindReduction: 20, // m/s average across forecast (from deltas: -35, -20, -10, ~0, 0)
}

// Drought analysis data from results_mideast.json
const droughtData = {
  region: "Middle East",
  forecastDate: "2022-01-01",
  control: {
    drought_cells: 24237,
    severity_score: 84337,
  },
  interventions: {
    electric_ionization: {
      precip_increase_pct: 15.7,
      cells_resolved: 21,
      cells_improved: 91,
      cells_worsened: 0,
      isBest: true,
    },
    glaciogenic_static: {
      precip_increase_pct: 4.1,
      cells_resolved: 7,
      cells_improved: 40,
      cells_worsened: 0,
      isBest: false,
    },
    hygroscopic_enhancement: {
      precip_increase_pct: 8.1,
      cells_resolved: 15,
      cells_improved: 56,
      cells_worsened: 0,
      isBest: false,
    },
    laser_induced_condensation: {
      precip_increase_pct: 6.7,
      cells_resolved: 12,
      cells_improved: 51,
      cells_worsened: 0,
      isBest: false,
    },
  },
}

// Mapping from strategy id to intervention key in droughtData
const strategyToInterventionKey: Record<string, string> = {
  "electric-ionization": "electric_ionization",
  "glaciogenic": "glaciogenic_static",
  "hygroscopic": "hygroscopic_enhancement",
  "laser-induced": "laser_induced_condensation",
}

const graphcastStrategies: Strategy[] = [
  {
    id: "electric-ionization",
    name: "Electric Ionization",
    description: "",
    icon: "cloud",
    isOptimal: false,
    kpis: {
      soilMoisture: "+12%",
      windPeak: "-20kts",
      confidence: "94%",
    },
    mitigationLevel: 85,
  },
  {
    id: "glaciogenic",
    name: "Glaciogenic",
    description: "",
    icon: "droplets",
    isOptimal: false,
    kpis: {
      soilMoisture: "+10%",
      windPeak: "-15kts",
      confidence: "89%",
    },
    mitigationLevel: 75,
  },
  {
    id: "hygroscopic",
    name: "Hygroscopic",
    description: "",
    icon: "wind",
    isOptimal: false,
    kpis: {
      soilMoisture: "+8%",
      windPeak: "-12kts",
      confidence: "85%",
    },
    mitigationLevel: 70,
  },
  {
    id: "laser-induced",
    name: "Laser Induced Condensation",
    description: "",
    icon: "cloud",
    isOptimal: false,
    kpis: {
      soilMoisture: "+5%",
      windPeak: "-10kts",
      confidence: "82%",
    },
    mitigationLevel: 65,
  },
]

const earth2studioStrategies: Strategy[] = [
  {
    id: "cloud-seeding",
    name: "Cloud Seeding",
    description: "",
    icon: "cloud",
    isOptimal: true,
    kpis: {
      soilMoisture: "+15%",
      windPeak: "-18kts",
      confidence: "91%",
    },
    mitigationLevel: 80,
    variants: [
      { id: "experimental", name: "Experimental" },
      { id: "katrina-control", name: "Katrina Control" },
      { id: "katrina-seeded", name: "Katrina Seeded" }
    ],
  },
]

export default function DashboardPage() {
  const [activeStrategy, setActiveStrategy] = useState("cloud-seeding")
  const [selectedModel, setSelectedModel] = useState<ModelType>("earth2studio")
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null)
  const [modelRun, setModelRun] = useState(false)
  const [checkedStrategies, setCheckedStrategies] = useState<Set<string>>(new Set())

  const strategies = selectedModel === "earth2studio" ? earth2studioStrategies : graphcastStrategies
  const current = strategies.find((s) => s.id === activeStrategy) ?? strategies[0]

  const handleSelect = useCallback((id: string) => {
    setActiveStrategy(id)
  }, [])

  const handleModelChange = useCallback((model: ModelType) => {
    setSelectedModel(model)
    // Reset to first strategy of the new model
    setActiveStrategy(model === "earth2studio" ? "cloud-seeding" : "electric-ionization")
    setModelRun(false)
    setCheckedStrategies(new Set())
  }, [])

  const handleCheckChange = useCallback((id: string, checked: boolean) => {
    setCheckedStrategies(prev => {
      const next = new Set(prev)
      if (checked) {
        next.add(id)
      } else {
        next.delete(id)
      }
      return next
    })
  }, [])

  const allStrategiesChecked = checkedStrategies.size === graphcastStrategies.length
  const onlyElectricIonization = checkedStrategies.size === 1 && checkedStrategies.has("electric-ionization")
  const onlyGlaciogenic = checkedStrategies.size === 1 && checkedStrategies.has("glaciogenic")
  const onlyHygroscopic = checkedStrategies.size === 1 && checkedStrategies.has("hygroscopic")
  const onlyLaserInduced = checkedStrategies.size === 1 && checkedStrategies.has("laser-induced")

  const getGraphcastVideoSrc = () => {
    if (onlyElectricIonization) return "/videos/control_vs_electric_ionization.mp4"
    if (onlyGlaciogenic) return "/videos/control_vs_glaciogenic_static.mp4"
    if (onlyHygroscopic) return "/videos/control_vs_hygroscopic_enhancement.mp4"
    if (onlyLaserInduced) return "/videos/control_vs_laser_induced_condensation.mp4"
    if (allStrategiesChecked) return "/videos/all_interventions_comparison.mp4"
    return null
  }

  // Get the active intervention data for single strategy view
  const getActiveIntervention = () => {
    if (checkedStrategies.size !== 1) return null
    const strategyId = Array.from(checkedStrategies)[0]
    const interventionKey = strategyToInterventionKey[strategyId]
    if (!interventionKey) return null
    return {
      data: droughtData.interventions[interventionKey as keyof typeof droughtData.interventions],
      name: graphcastStrategies.find(s => s.id === strategyId)?.name || strategyId,
    }
  }

  const activeIntervention = getActiveIntervention()

  return (
    <div className="flex flex-col h-screen bg-background">
      <DashboardHeader
        selectedModel={selectedModel}
        onModelChange={handleModelChange}
      />

      <div className="flex flex-1 min-h-0">
        <StrategyCards
          strategies={strategies}
          activeId={activeStrategy}
          onSelect={handleSelect}
          sidebarTitle={selectedModel === "earth2studio" ? "Method" : "Strategies"}
          selectedVariant={selectedVariant}
          onVariantSelect={setSelectedVariant}
          checkboxMode={selectedModel === "graphcast"}
          modelRun={modelRun}
          onModelRun={() => setModelRun(true)}
          checkedStrategies={checkedStrategies}
          onCheckChange={handleCheckChange}
        />

        <div className="flex flex-col flex-1 min-w-0">
          {selectedModel === "earth2studio" ? (
            selectedVariant === "experimental" ? (
              <>
                <DataSummaryPanel
                  cycloneExperimentData={cycloneExperimentData}
                />
                <VideoComparison
                  controlSrc="/videos/control_forecast.mov"
                  seededSrc="/videos/seeded_forecast.mov"
                  strategyName={current.name}
                  playbackRate={1}
                />
              </>
            ) : selectedVariant === "katrina-control" ? (
              <>
                <DataSummaryPanel
                  katrinaData={katrinaData}
                  katrinaVariant="control"
                />
                <SingleVideoView
                  src="/videos/control_forecast_katrina.mp4"
                  strategyName="Katrina Control"
                  modelName="Earth2Studio"
                  playbackRate={0.25}
                />
              </>
            ) : selectedVariant === "katrina-seeded" ? (
              <>
                <DataSummaryPanel
                  katrinaData={katrinaData}
                  katrinaVariant="seeded"
                />
                <SingleVideoView
                  src="/videos/seeded_forecast_katrina.mp4"
                  strategyName="Katrina Seeded"
                  modelName="Earth2Studio"
                  playbackRate={0.25}
                />
              </>
            ) : (
              <div className="flex-1 bg-white" />
            )
          ) : modelRun && getGraphcastVideoSrc() ? (
            <>
              <DataSummaryPanel
                region={droughtData.region}
                forecastDate={droughtData.forecastDate}
                control={droughtData.control}
                intervention={activeIntervention?.data}
                interventionName={activeIntervention?.name}
                allInterventions={allStrategiesChecked ? droughtData.interventions : undefined}
                showAllComparison={allStrategiesChecked}
              />
              <SingleVideoView
                src={getGraphcastVideoSrc()!}
                strategyName={current.name}
                modelName="GraphCast"
              />
            </>
          ) : (
            <div className="flex-1 bg-white" />
          )}
        </div>
      </div>
    </div>
  )
}
