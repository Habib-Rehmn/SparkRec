# ADR 002: Next.js frontend with FastAPI

Status: accepted, 2026-10-05. Implementation milestone: M3.

The user selected Next.js instead of the originally proposed Streamlit dashboard.
Use Next.js with TypeScript for user selection, movie search, previous ratings,
recommendations, fallback labels, snapshot age, and model comparison displays.
FastAPI exposes read-only data from validated, versioned SQLite snapshots.

The frontend and API use separate Docker images and Compose services. Neither
service starts Spark or trains models during a request. M2 evaluation outputs
remain separate from the all-history fit and published serving snapshot added in M3.

This adds a JavaScript toolchain and an API/frontend contract to maintain, while
allowing a dedicated web interface. Authentication and live rating updates remain
outside the initial demo scope.
