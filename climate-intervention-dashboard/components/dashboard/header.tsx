export type ModelType = "earth2studio" | "graphcast"

interface DashboardHeaderProps {
  selectedModel: ModelType
  onModelChange: (model: ModelType) => void
}

function HurricaneLogo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Outer spiral arm */}
      <path
        d="M12 2C6.5 2 2 6.5 2 12c0 1.5.3 2.9.9 4.2"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      {/* Middle spiral arm */}
      <path
        d="M5.5 6.5C7.5 4.5 10 3.5 12 4c2 .5 3.5 2 4 4 .5 2-.5 4.5-2.5 6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      {/* Inner spiral arm */}
      <path
        d="M9 9c1-1 2.5-1.5 4-1s2.5 1.5 2.5 3-1 2.5-2.5 3"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      {/* Hurricane eye */}
      <circle
        cx="12"
        cy="12"
        r="1.5"
        fill="currentColor"
      />
      {/* Trailing spiral outward */}
      <path
        d="M14 16c2 1 4 1.5 6 1s3.5-2 4-4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function DashboardHeader({ selectedModel, onModelChange }: DashboardHeaderProps) {
  return (
    <header className="flex items-center h-12 px-5 border-b border-border bg-background">
      <div className="flex items-center gap-3">
        <HurricaneLogo className="h-5 w-5 text-primary" />
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
