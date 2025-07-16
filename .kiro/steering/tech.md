---
inclusion: always
---

# Technical Stack & Development Standards

## Core Technology Stack

### Backend (Python)

- **FastAPI + Uvicorn**: Async API framework for high-performance trading operations
- **SQLAlchemy + Alembic**: ORM with migration support for financial data integrity
- **CCXT**: Multi-exchange trading library for unified API access
- **ML Stack**: pandas, numpy, scikit-learn, LightGBM for strategy development
- **Backtesting**: Custom backtesting framework for strategy validation
- **Optimization**: DEAP (genetic algorithms), scikit-optimize (Bayesian optimization)

### Frontend (TypeScript/React)

- **Next.js 15**: Full-stack React framework with App Router
- **Radix UI + Tailwind**: Accessible components with custom design system
- **React Hook Form + Zod**: Type-safe form handling and validation
- **Recharts**: Financial data visualization and charting
- **React Context**: State management for trading data

## Development Standards

### Python Code Patterns

- **MUST use async/await** for all I/O operations (database, API calls)
- **MUST use dependency injection** via FastAPI's Depends system
- **MUST implement repository pattern** for data access layers
- **MUST use Pydantic models** for request/response validation
- **MUST follow service layer pattern** for business logic separation

### TypeScript/React Patterns

- **MUST use TypeScript strict mode** with proper type definitions
- **MUST create custom hooks** for shared stateful logic
- **MUST use React.memo** for expensive component re-renders
- **MUST implement error boundaries** for trading operation failures
- **MUST use Zod schemas** for runtime type validation

### Code Quality Standards

- **Python**: Run `black`, `isort`, `flake8`, `mypy` before commits
- **TypeScript**: Use ESLint rules, ensure no TypeScript errors
- **MUST write unit tests** for all business logic functions
- **MUST mock external APIs** in tests (exchanges, databases)
- **MUST use type hints** in all Python function signatures

## Development Workflows

### Backend Development

```bash
# Environment setup (run once)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Daily development cycle
python main.py                    # Start dev server
pytest tests/                     # Run tests
black . && isort . && flake8 .   # Format and lint
mypy .                           # Type checking

# Database operations
alembic upgrade head              # Apply migrations
alembic revision --autogenerate -m "description"  # Create migration
```

### Frontend Development

```bash
# Environment setup (run once)
npm install

# Daily development cycle
npm run dev                      # Start dev server
npm test                        # Run tests
npm run lint                    # ESLint checking
npm run build                   # Verify production build

# Testing workflows
npm run test:watch              # Watch mode for TDD
npm run test:coverage           # Coverage reports
```

## Configuration Management

### Environment-Based Configuration

- **Backend**: Use `python-dotenv` for environment variables
- **Database**: SQLAlchemy connection strings via environment
- **API Keys**: Store in environment, never in code or logs
- **Frontend**: Next.js environment variables with `NEXT_PUBLIC_` prefix

### Development vs Production

- **Development**: SQLite database, mock trading APIs
- **Production**: PostgreSQL, real exchange connections
- **MUST use different API keys** per environment
- **MUST validate configuration** on application startup

## Testing Strategy

### Backend Testing

- **Unit Tests**: pytest with async support for business logic
- **Integration Tests**: Test database operations with test database
- **API Tests**: FastAPI TestClient for endpoint validation
- **MUST mock CCXT** exchange calls in tests
- **MUST use fixtures** for test data setup

### Frontend Testing

- **Component Tests**: React Testing Library for UI components
- **Hook Tests**: Custom hook testing with renderHook
- **Integration Tests**: Full user flow testing
- **MUST mock API calls** using MSW or similar
- **MUST test error states** and loading states

## Performance Guidelines

### Backend Performance

- **Use connection pooling** for database and exchange APIs
- **Implement caching** for frequently accessed market data
- **Use background tasks** for non-critical operations
- **Monitor async task queues** to prevent blocking

### Frontend Performance

- **Use React.lazy** for code splitting on route level
- **Implement virtual scrolling** for large data tables
- **Cache API responses** using React Query or SWR
- **Optimize bundle size** with Next.js analyzer

## Required Versions

- **Python**: 3.10+ (required for modern async features)
- **Node.js**: 18+ (Next.js 15 compatibility)
- **PostgreSQL**: 13+ (for production deployments)
