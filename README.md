# MARSA

Multi-Agent ReSearch Assistant.

## Setup

### Prerequisites

- Python 3.12+
- Node.js 20+
- API keys for:
  - [Anthropic Claude](https://console.anthropic.com/)
  - [OpenAI](https://platform.openai.com/api-keys) (for embeddings)
  - [Tavily](https://tavily.com/)

### Quick Start

1. **Clone and install dependencies:**

   ```bash
   make setup
   ```

2. **Configure API keys:**

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Run the application:**

   ```bash
   # Terminal 1: Backend
   make run-backend
   
   # Terminal 2: Frontend
   make run-frontend
   ```

### Development Commands

- `make setup` - Install all dependencies
- `make test` - Run tests
- `make lint` - Lint code
- `make docker` - Run with Docker Compose

### CI/CD

The project uses GitHub Actions for automated testing and linting. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

**Note:** This project is in active development.
