# 🏛️ Logical-Functional Architecture

graph TB
    %% External Interfaces
    subgraph "🌐 External Interfaces"
        API[🔌 REST API<br/>FastAPI]
        WEB[🖥️ Web Interface<br/>Dashboard]
        CLI[⚡ CLI Tools<br/>Scripts]
    end

    %% Core Application Layer
    subgraph "🎯 Core Application Layer"
        APP[🚀 Application<br/>Orchestrator]
        MIDDLEWARE[⚙️ Middleware<br/>Processing]
        DEPS[🔗 Dependency<br/>Injection]
    end

    %% Agent Framework Layer
    subgraph "🤖 AI Agent Framework"
        BASE[📋 Base Agent<br/>Framework]
        
        subgraph "🔍 Discovery Agents"
            SOCIAL[📱 Social Intelligence<br/>Agent]
            SEARCH[🔎 Search Strategy<br/>Generator]
        end
        
        subgraph "📄 Extraction Agents"
            EXTRACT[🎯 Content Extractor<br/>Orchestrator]
            ANALYZER[🧠 Website Analyzer<br/>Playwright]
            SYNTH[⚗️ Data Synthesizer<br/>AI-Powered]
        end
        
        subgraph "🧠 Intelligence Agents"
            MARKET[📊 Market Intelligence<br/>Agent]
            COMPETITIVE[⚔️ Competitive Analysis<br/>Agent]
            TRENDS[📈 Trend Analysis<br/>Agent]
            ORCHESTRA[🎼 Intelligence<br/>Orchestrator]
        end
    end

    %% Service Layer
    subgraph "🔧 Service Layer"
        AICLIENT[🤖 AI Async Client<br/>Cost Optimization]
        DATABASE[🗄️ Database Service<br/>PostgreSQL]
        CACHE[⚡ Cache Service<br/>Redis]
        STORAGE[💾 Storage Service<br/>File Management]
        NOTIFY[📧 Notification<br/>Service]
    end

    %% Background Processing
    subgraph "⚙️ Background Processing"
        CELERY[🔄 Celery Workers<br/>Task Queue]
        SCHEDULER[⏰ Task Scheduler<br/>Cron Jobs]
        MONITOR[📊 Health Monitor<br/>Metrics]
    end

    %% Data Layer
    subgraph "💾 Data Layer"
        MODELS[📋 Data Models<br/>Pydantic/SQLAlchemy]
        REPOS[🏛️ Repositories<br/>Data Access]
        MIGRATIONS[🔄 Database<br/>Migrations]
    end

    %% External Services
    subgraph "🌍 External Services"
        OLLAMA[🦙 Ollama<br/>Local AI]
        HUGGINGFACE[🤗 HuggingFace<br/>Cloud AI]
        GROQ[⚡ Groq<br/>Fast AI]
        OPENAI[🧠 OpenAI<br/>Advanced AI]
        PLAYWRIGHT[🎭 Playwright<br/>Browser Automation]
    end

    %% Connections - External Interfaces
    API --> APP
    WEB --> APP
    CLI --> APP

    %% Connections - Core Layer
    APP --> MIDDLEWARE
    APP --> DEPS
    MIDDLEWARE --> BASE

    %% Connections - Agent Framework
    BASE --> SOCIAL
    BASE --> EXTRACT
    BASE --> MARKET
    BASE --> COMPETITIVE
    BASE --> TRENDS
    
    EXTRACT --> ANALYZER
    EXTRACT --> SYNTH
    
    ORCHESTRA --> MARKET
    ORCHESTRA --> COMPETITIVE
    ORCHESTRA --> TRENDS

    %% Connections - Services
    BASE --> AICLIENT
    APP --> DATABASE
    APP --> CACHE
    APP --> STORAGE
    APP --> NOTIFY

    %% Connections - Background Processing
    APP --> CELERY
    CELERY --> SCHEDULER
    CELERY --> MONITOR

    %% Connections - Data Layer
    BASE --> MODELS
    DATABASE --> REPOS
    REPOS --> MODELS
    DATABASE --> MIGRATIONS

    %% Connections - External Services
    AICLIENT --> OLLAMA
    AICLIENT --> HUGGINGFACE
    AICLIENT --> GROQ
    AICLIENT --> OPENAI
    ANALYZER --> PLAYWRIGHT

    %% Styling
    classDef interface fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef core fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef agent fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef service fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef data fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef external fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class API,WEB,CLI interface
    class APP,MIDDLEWARE,DEPS core
    class BASE,SOCIAL,EXTRACT,ANALYZER,SYNTH,MARKET,COMPETITIVE,TRENDS,ORCHESTRA agent
    class AICLIENT,DATABASE,CACHE,STORAGE,NOTIFY,CELERY,SCHEDULER,MONITOR service
    class MODELS,REPOS,MIGRATIONS data
    class OLLAMA,HUGGINGFACE,GROQ,OPENAI,PLAYWRIGHT external