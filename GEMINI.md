# GEMINI.md - Project Context

- 最初の指示を Serena MCP で確認してください
- ライブラリの事は Context7 でドキュメントを調べてください
- TDD で開発してください
- ユーザーはコードをすし詰め状態で書くことを好まない
- ユーザーとの会話は、必ず日本語で行う。
- コードスタイルは可読性を第一に考える
- ユーザーは複雑な課題を小さなステップに分けて解決することを好む。
- ユーザーはコードの可読性を重視しており、整ったインデント、命名規則、適切なコメントを求めている。
- ユーザーとの会話は、必ず日本語で行う。専門用語は必要に応じて補足説明を行う。
- ユーザーは実装前に目的、影響、リスク、代替案を深く検討することを重視している。
- ユーザーは、エージェントが勝手にコミットしないことを希望しています。コミットはユーザー自身が行います。
- ユーザーの OS は Windows です。

This document provides essential context for the `trading` project, designed to help AI agents understand its structure, purpose, and conventions.

## Project Overview

This project is a comprehensive platform for researching, developing, and optimizing cryptocurrency trading strategies. It features a web-based user interface for interacting with a powerful backend that handles data processing, backtesting, and machine learning model management.

### Architecture

The project is a monorepo containing two main parts:

1.  **`frontend`**: A [Next.js](https://nextjs.org/) application written in [TypeScript](https://www.typescriptlang.org/). It provides the user interface for data visualization, backtest configuration, and viewing ML model results.
2.  **`backend`**: An API built with [Python](https://www.python.org/) and [FastAPI](https://fastapi.tiangolo.com/). It manages data collection, strategy backtesting, database interactions, and machine learning workflows.

### Key Technologies

- **Frontend**:

  - Framework: Next.js
  - Language: TypeScript
  - Styling: Tailwind CSS
  - UI Components: shadcn/ui, Radix UI
  - Charting: Recharts
  - Testing: Jest

- **Backend**:
  - Framework: FastAPI
  - Language: Python
  - Database ORM: SQLAlchemy
  - Database Migrations: Alembic
  - Data Analysis: Pandas, NumPy
  - ML/Optimization: scikit-learn, LightGBM, Optuna, DEAP
  - Crypto Exchange API: CCXT
  - Testing: Pytest

## Building and Running

### Backend (`/backend`)

1.  **Install Dependencies**: It's recommended to use a virtual environment.

    ```bash
    # Navigate to the backend directory
    cd backend

    # Install dependencies
    pip install -e .[test,dev]
    ```

2.  **Run Development Server**:

    ```bash
    # This will start the FastAPI server with auto-reload
    uvicorn app.main:app --reload
    ```

    Alternatively, you can run the main script:

    ```bash
    python main.py
    ```

3.  **Run Tests**:
    ```bash
    pytest
    ```

### Frontend (`/frontend`)

1.  **Install Dependencies**:

    ```bash
    # Navigate to the frontend directory
    cd frontend

    # Install dependencies
    npm install
    ```

2.  **Run Development Server**:

    ```bash
    npm run dev
    ```

    The application will be available at `http://localhost:3000`.

3.  **Build for Production**:

    ```bash
    npm run build
    ```

4.  **Run Production Server**:

    ```bash
    npm run start
    ```

5.  **Run Tests**:
    ```bash
    npm run test
    ```

## Development Conventions

### Backend

- **Code Style**: Code is formatted using `black` and `isort`.
- **Static Analysis**: `mypy` is used for type checking, and `flake8` for linting. Please adhere to their standards.
- **Database**: Database schema changes are managed through `alembic` migrations. When making model changes, generate a new migration script.

### Frontend

- **Linting**: `ESLint` is used for code quality. Run `npm run lint` to check for issues.
- **Component Library**: The UI is built upon `shadcn/ui`, which uses Radix UI primitives. When creating new UI elements, try to compose them from existing components in `frontend/components/ui`.
- **Styling**: Use Tailwind CSS utility classes for styling. Avoid writing custom CSS files where possible.
