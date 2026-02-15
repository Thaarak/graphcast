# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Climate Intervention Dashboard ("Morro") is a Next.js visualization app for comparing climate intervention strategies. It displays forecast videos from two AI weather models (Earth2Studio and GraphCast) with side-by-side control vs. intervention comparisons.

## Development Commands

```bash
pnpm dev      # Start dev server with Turbopack (localhost:3000)
pnpm build    # Production build
pnpm lint     # Run ESLint
```

## Architecture

### Page Structure
- `app/page.tsx` - Main dashboard page (client component)
- Two model modes: **Earth2Studio** (single strategy with variants) and **GraphCast** (multi-select checkbox comparison)

### Component Organization
```
components/
├── dashboard/          # Feature components
│   ├── header.tsx          # Model switcher (Earth2Studio/GraphCast)
│   ├── strategy-cards.tsx  # Left sidebar with intervention strategies
│   ├── data-summary-panel.tsx  # Metrics header (wind reduction, precipitation, cells resolved)
│   ├── video-comparison.tsx    # Synced dual-video player (control vs seeded)
│   └── single-video-view.tsx   # Single video with overlay labels
└── ui/                 # Shadcn UI primitives (60+ components)
```

### Data Flow
1. User selects model (Earth2Studio/GraphCast) in header
2. Strategy cards show available interventions with KPIs
3. For Earth2Studio: select variant (experimental, katrina-control, katrina-seeded)
4. For GraphCast: checkbox multi-select strategies, then click "Model" button
5. DataSummaryPanel shows metrics; video player shows comparison

### Key Data Structures (in `app/page.tsx`)
- `droughtData`: Middle East drought intervention results (precipitation %, cells resolved/improved)
- `katrinaData`: Hurricane Katrina wind speed reduction metrics
- `cycloneExperimentData`: Cyclone simulation average wind reduction
- `graphcastStrategies` / `earth2studioStrategies`: Strategy definitions with KPIs

### Video Assets
Pre-recorded comparison videos in `public/videos/`:
- `control_vs_{intervention}.mp4` - Individual strategy comparisons
- `all_interventions_comparison.mp4` - Multi-strategy view
- `control_forecast_katrina.mp4` / `seeded_forecast_katrina.mp4` - Katrina experiment

## Styling

- Tailwind CSS with dark mode (class-based)
- CSS variables defined in `app/globals.css` for theming
- Uses `cn()` utility from `lib/utils.ts` for class merging

## Tech Stack

- Next.js 16.1.6 with App Router and Turbopack
- React 19.2.3, TypeScript 5.7.3
- Radix UI primitives via Shadcn
- Lucide icons
- Path alias: `@/*` maps to project root
