---
inclusion: always
---

# Technical Stack & Development Standards

## Core Technology Stack

**Backend**: FastAPI + SQLAlchemy + CCXT + pandas/numpy/scikit-learn  
**Frontend**: Next.js 15 + Radix UI + Tailwind + Recharts  
**Testing**: pytest (backend) + Jest/RTL (frontend)

## Critical Financial Code Requirements

- **NEVER use `float` for financial calculations** - Always use `Decimal` type
- **Use 8 decimal precision** for cryptocurrency pairs
- **Implement `ROUND_HALF_UP`** for all financial rounding operations
- **Validate all financial calculations** with unit tests using known expected results

```python
from decimal import Decimal, ROUND_HALF_UP

# Correct financial calculation pattern
price = Decimal('0.12345678')  # Never float
amount = price.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
```

## Mandatory Code Patterns

### Python (Backend)

- **Always use `async/await`** for I/O operations (database, CCXT, APIs)
- **Use dependency injection** with FastAPI's `Depends()` system
- **Type hints required** on all function signatures
- **Pydantic models** for request/response validation

```python
@router.post("/strategy", response_model=StrategyResponse)
async def create_strategy(
    strategy: StrategyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> StrategyResponse:
    try:
        # Implementation with proper error handling
        pass
    except Exception as e:
        # Log with correlation ID for tracing
        raise HTTPException(status_code=500, detail="Strategy creation failed")
```

### TypeScript (Frontend)

- **TypeScript strict mode** with complete type definitions
- **Custom hooks** for shared stateful logic
- **Zod schemas** for runtime validation
- **Proper spacing** in control structures (if statements, loops)

```typescript
const { data, error, isLoading } = useSWR<PortfolioData>(
  "/api/portfolio",
  fetcher,
  { refreshInterval: 1000 }
);

if (error) {
  // Handle error state with user feedback
}
```

## Performance & Security Standards

### Performance Targets

- Market data processing: **< 100ms**
- Strategy signal generation: **< 500ms**
- Portfolio updates: **< 1 second**

### Required Implementations

- **Connection pooling** for database and exchange APIs
- **Caching** for frequently accessed market data
- **Circuit breakers** for API rate limits
- **Structured logging** with correlation IDs

### Security Requirements

- **Environment variables** for all configuration
- **Separate API keys** per environment and exchange
- **Input validation** before processing trading operations
- **Audit logging** for all portfolio changes
- **Never log API keys** in responses or error messages

## Development Workflow

```bash
# Pre-commit checks
# Python
black . && isort . && flake8 . && mypy .

# TypeScript
npm run lint && npm run type-check && npm test

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"
```

## Testing Requirements

- **Mock all external APIs** (CCXT, databases) in tests
- **Test financial calculations** with known expected results
- **Test error states** and loading states in UI components
- **Test concurrent operations** for race conditions
- **Use fixtures** for consistent test data setup
