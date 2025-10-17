# Project Overview

This is a trading application with a Python/FastAPI backend and a Next.js/React frontend.

The backend is a sophisticated trading API that uses a variety of libraries for data analysis, database management, and machine learning. It appears to be designed for algorithmic trading, with features for backtesting, genetic algorithms, and hyperparameter optimization.

The frontend is a Next.js application that provides a user interface for interacting with the trading backend. It uses a modern UI stack with Radix UI, Tailwind CSS, and Recharts for data visualization.

.最初の指示で serena を使ってください
.実装作業に入る前に serena の think を使って思考してください.
.実装する前に serena のメモリも確認してください
.返答は日本語
.わからなければウェブ検索を行い調べてください
.ライブラリは context7 ドキュメントを調べてください
.TDD で開発してください

## Key Technologies

### Backend

- **Framework:** FastAPI
- **Database:** PostgreSQL (inferred from `psycopg2-binary`)
- **ORM:** SQLAlchemy
- **Database Migrations:** Alembic
- **Data Analysis:** pandas, numpy, pandas_ta
- **Machine Learning:** scikit-learn, lightgbm, pytorch-tabnet, tsfresh, autofeat
- **Hyperparameter Optimization:** optuna
- **Genetic Algorithms:** deap
- **API Client:** httpx
- **Code Style:** Black, isort
- **Type Checking:** Mypy

### Frontend

- **Framework:** Next.js (React)
- **Language:** TypeScript
- **UI Components:** Radix UI, lucide-react
- **Styling:** Tailwind CSS
- **Charting:** Recharts
- **Linting:** ESLint
- **Testing:** Jest, React Testing Library

## Project Structure

### Backend (`backend/`)

```
backend/
├── alembic/              # Database migration scripts
├── app/                  # Main application code
│   ├── api/              # API endpoints
│   ├── config/           # Application configuration
│   ├── services/         # Business logic
│   └── utils/            # Utility functions
├── database/             # Database connection and models
│   └── repositories/     # Data access layer
├── models/               # Pydantic models (likely)
├── scripts/              # Standalone scripts
└── tests/                # Tests
```

### Frontend (`frontend/`)

```
frontend/
├── app/                  # Next.js app directory
│   ├── (pages)/          # Page components
│   ├── globals.css       # Global styles
│   └── layout.tsx        # Root layout
├── components/           # Reusable UI components
├── constants/            # Application constants
├── hooks/                # Custom React hooks
├── lib/                  # Library functions
├── types/                # TypeScript type definitions
└── utils/                # Utility functions
```

## Building and Running

### Backend

To install the backend dependencies, run the following command in the `backend` directory:

```bash
pip install -e .[test,dev]
```

To run the backend server, run the following command in the `backend` directory:

```bash
uvicorn app.main:app --reload
```

### Frontend

To install the frontend dependencies, run the following command in the `frontend` directory:

```bash
npm install
```

To run the frontend development server, run the following command in the `frontend` directory:

```bash
npm run dev
```

## Development Conventions

### Backend

The backend code follows the Black code style and uses isort for import sorting. Mypy is used for static type checking. These are enforced via the `pyproject.toml` file.

### Frontend

The frontend code uses ESLint for linting, configured in `.eslintrc.json`. Code formatting is likely handled by Prettier, as is common in Next.js projects, but no `.prettierrc` file is present. Testing is done with Jest and React Testing Library, configured in `jest.config.js` and `jest.setup.js`.
