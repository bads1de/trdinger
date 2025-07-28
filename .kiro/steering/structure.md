---
inclusion: always
---

# Project Structure & Organization

## File Placement Rules

### Backend (`backend/`)

- **API Routes**: Place in `backend/app/api/` with descriptive names (e.g., `trading_strategies.py`)
- **Business Logic**: Core logic goes in `backend/app/core/` organized by domain
- **Database Models**: SQLAlchemy models in `backend/models/` with one model per file
- **Configuration**: Environment-specific configs in `backend/app/config/`
- **Data Collection**: Market data collectors in `backend/data_collector/`
- **Scripts**: Maintenance and utility scripts in `backend/scripts/`
- **Tests**: Mirror app structure in `backend/tests/` (e.g., `tests/api/test_trading_strategies.py`)

### Frontend (`frontend/`)

- **Pages**: Next.js App Router pages in `frontend/app/` following route structure
- **Components**: Reusable UI components in `frontend/components/` with kebab-case names
- **Hooks**: Custom React hooks in `frontend/hooks/` prefixed with `use`
- **Types**: TypeScript definitions in `frontend/types/` organized by domain
- **Utils**: Helper functions in `frontend/utils/` with descriptive names
- **Constants**: Application constants in `frontend/constants/`
- **Tests**: Component tests in `frontend/__tests__/` mirroring component structure

## Architecture Patterns

### Backend Patterns

- **Layered Architecture**: API → Service → Repository → Database
- **Dependency Injection**: Use FastAPI's `Depends()` for all dependencies
- **Repository Pattern**: Abstract database access behind repository interfaces
- **Service Layer**: Business logic separated from API handlers
- **Configuration Management**: Environment-based settings with validation

### Frontend Patterns

- **Component Composition**: Build complex UIs from simple, reusable components
- **Custom Hooks**: Extract stateful logic into reusable hooks
- **Type-Safe APIs**: Use TypeScript interfaces for all backend communication
- **File-Based Routing**: Follow Next.js App Router conventions strictly

## Naming Conventions

### Python

- **Functions/Variables**: `snake_case` (e.g., `calculate_portfolio_value`)
- **Classes**: `PascalCase` (e.g., `TradingStrategy`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_POSITION_SIZE`)
- **Files**: `snake_case.py` (e.g., `trading_strategy.py`)

### TypeScript

- **Functions/Variables**: `camelCase` (e.g., `calculatePortfolioValue`)
- **Components/Types**: `PascalCase` (e.g., `TradingDashboard`, `PortfolioData`)
- **Files**: `kebab-case.tsx` for components, `camelCase.ts` for utilities

### Directories

- Use lowercase with hyphens or underscores consistently within each section
- Backend: `snake_case` directories
- Frontend: `kebab-case` directories

## Import Organization

### Python

```python
# Standard library imports
import asyncio
from datetime import datetime

# Third-party imports
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

# Local imports
from app.core.trading import TradingEngine
from app.models.portfolio import Portfolio
```

### TypeScript

```typescript
// External libraries
import React from "react";
import { NextPage } from "next";

// Internal utilities
import { formatCurrency } from "@/utils/formatting";

// Components
import { TradingChart } from "@/components/trading-chart";

// Types
import type { PortfolioData } from "@/types/portfolio";
```

## Directory Structure Rules

### When to Create New Directories

- **Backend**: Create new directories in `app/` when you have 3+ related modules
- **Frontend**: Create new component directories when you have shared sub-components
- **Tests**: Always mirror the structure of the code being tested

### File Organization

- **One class per file** for models and major components
- **Group related functions** in utility modules
- **Separate concerns** - don't mix API logic with business logic
- **Co-locate tests** with the code they test when possible

## Testing Structure

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test API endpoints and database interactions
- **Component Tests**: Test React components with React Testing Library
- **E2E Tests**: Test critical user flows across the entire application
