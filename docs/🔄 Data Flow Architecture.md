# 🔄 Data Flow Architecture

# flowchart TD

    %% Input Layer
    REQUEST[📥 User Request<br/>Discovery/Extraction/Intelligence]
    
    %% Processing Pipeline
    subgraph "🔄 Processing Pipeline"
        VALIDATE[✅ Request Validation<br/>& Authentication]
        ROUTE[🚏 Request Routing<br/>& Load Balancing]
        QUEUE[📋 Task Queuing<br/>Background Processing]
    end
    
    %% Agent Processing
    subgraph "🤖 Agent Processing Layer"
        DISCOVER[🔍 Discovery<br/>Web/Social Search]
        EXTRACT[📄 Extraction<br/>Content Analysis]
        INTELLIGENCE[🧠 Intelligence<br/>Business Analysis]
    end
    
    %% AI Processing
    subgraph "🧠 AI Processing Layer"
        AI_ROUTER[🎯 AI Provider<br/>Router]
        AI_COST[💰 Cost<br/>Optimizer]
        AI_CACHE[⚡ Response<br/>Cache]
    end
    
    %% Data Processing
    subgraph "💾 Data Processing Layer"
        VALIDATE_DATA[✅ Data<br/>Validation]
        TRANSFORM[🔄 Data<br/>Transformation]
        ENRICH[⭐ Data<br/>Enrichment]
    end
    
    %% Storage Layer
    subgraph "🗄️ Storage Layer"
        PRIMARY_DB[(🗄️ Primary Database<br/>PostgreSQL)]
        CACHE_DB[(⚡ Cache<br/>Redis)]
        FILE_STORE[(📁 File Storage<br/>Local/Cloud)]
    end
    
    %% Output Layer
    RESPONSE[📤 Response<br/>Structured Data]
    
    %% Flow Connections
    REQUEST --> VALIDATE
    VALIDATE --> ROUTE
    ROUTE --> QUEUE
    QUEUE --> DISCOVER
    QUEUE --> EXTRACT
    QUEUE --> INTELLIGENCE
    
# 🔄 Data Flow Architecture  

    DISCOVER --> AI_ROUTER
    EXTRACT --> AI_ROUTER
    INTELLIGENCE --> AI_ROUTER
    
    AI_ROUTER --> AI_COST
    AI_COST --> AI_CACHE
    AI_CACHE --> VALIDATE_DATA
    
    VALIDATE_DATA --> TRANSFORM
    TRANSFORM --> ENRICH
    ENRICH --> PRIMARY_DB
    ENRICH --> CACHE_DB
    ENRICH --> FILE_STORE
    
    PRIMARY_DB --> RESPONSE
    CACHE_DB --> RESPONSE
    
    %% Styling
    classDef input fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef agent fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef ai fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef storage fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    classDef output fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    
    class REQUEST input
    class VALIDATE,ROUTE,QUEUE process
    class DISCOVER,EXTRACT,INTELLIGENCE agent
    class AI_ROUTER,AI_COST,AI_CACHE ai
    class VALIDATE_DATA,TRANSFORM,ENRICH data
    class PRIMARY_DB,CACHE_DB,FILE_STORE storage
    class RESPONSE output