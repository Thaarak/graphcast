import { Activity } from "lucide-react"

export type ModelType = "earth2studio" | "graphcast"

interface DashboardHeaderProps {
  selectedModel: ModelType
  onModelChange: (model: ModelType) => void
}

export function DashboardHeader({ selectedModel, onModelChange }: DashboardHeaderProps) {
  return (
    <header className="flex items-center h-12 px-5 border-b border-border bg-background">
      <div className="flex items-center gap-3">
        <Activity className="h-4 w-4 text-primary" />
        <h1 className="text-sm font-semibold tracking-tight text-foreground">
          Morro
        </h1>
      </div>

      <div className="flex-1 flex items-center justify-center">
        <div className="flex items-center bg-muted rounded-md p-0.5">
          <button
            onClick={() => onModelChange("earth2studio")}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              selectedModel === "earth2studio"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Earth2Studio
          </button>
          <button
            onClick={() => onModelChange("graphcast")}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              selectedModel === "graphcast"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            GraphCast
          </button>
        </div>
      </div>
    </header>
  )
}
