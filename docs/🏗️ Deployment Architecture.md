# 🏗️ Deployment Architecture

graph TB
    %% Load Balancer Layer
    subgraph "🌐 Load Balancer Layer"
        LB[⚖️ Load Balancer<br/>Nginx/HAProxy]
        CDN[🌍 CDN<br/>Static Assets]
    end
    
    %% Application Layer
    subgraph "🚀 Application Layer"
        API1[🔌 API Instance 1<br/>FastAPI]
        API2[🔌 API Instance 2<br/>FastAPI]
        API3[🔌 API Instance 3<br/>FastAPI]
        
        WEB1[🖥️ Web Instance 1<br/>Dashboard]
        WEB2[🖥️ Web Instance 2<br/>Dashboard]
    end
    
    %% Worker Layer
    subgraph "⚙️ Worker Layer"
        WORKER1[👷 Worker 1<br/>Discovery]
        WORKER2[👷 Worker 2<br/>Extraction]
        WORKER3[👷 Worker 3<br/>Intelligence]
        WORKER4[👷 Worker 4<br/>Analytics]
    end
    
    %% Queue Layer
    subgraph "📋 Message Queue Layer"
        REDIS_QUEUE[(📋 Redis<br/>Task Queue)]
        REDIS_CACHE[(⚡ Redis<br/>Cache)]
    end
    
    %% Database Layer
    subgraph "🗄️ Database Layer"
        POSTGRES_PRIMARY[(🗄️ PostgreSQL<br/>Primary)]
        POSTGRES_REPLICA[(🗄️ PostgreSQL<br/>Read Replica)]
    end
    
    %% Monitoring Layer
    subgraph "📊 Monitoring Layer"
        PROMETHEUS[📊 Prometheus<br/>Metrics]
        GRAFANA[📈 Grafana<br/>Dashboards]
        ALERTMANAGER[🚨 AlertManager<br/>Notifications]
    end
    
    %% External Services
    subgraph "🌍 External AI Services"
        OLLAMA_LOCAL[🦙 Ollama<br/>Local Deployment]
        AI_CLOUD[☁️ Cloud AI<br/>HF/Groq/OpenAI]
    end
    
    %% User Traffic
    USERS[👥 Users] --> LB
    
    %% Load Balancer Routing
    LB --> API1
    LB --> API2
    LB --> API3
    LB --> WEB1
    LB --> WEB2
    
    %% API to Services
    API1 --> REDIS_QUEUE
    API2 --> REDIS_QUEUE
    API3 --> REDIS_QUEUE
    
    API1 --> REDIS_CACHE
    API2 --> REDIS_CACHE
    API3 --> REDIS_CACHE
    
    API1 --> POSTGRES_PRIMARY
    API2 --> POSTGRES_REPLICA
    API3 --> POSTGRES_REPLICA
    
    %% Workers to Services
    WORKER1 --> REDIS_QUEUE
    WORKER2 --> REDIS_QUEUE
    WORKER3 --> REDIS_QUEUE
    WORKER4 --> REDIS_QUEUE
    
    WORKER1 --> POSTGRES_PRIMARY
    WORKER2 --> POSTGRES_PRIMARY
    WORKER3 --> POSTGRES_PRIMARY
    WORKER4 --> POSTGRES_PRIMARY
    
    %% Workers to AI Services
    WORKER1 --> OLLAMA_LOCAL
    WORKER2 --> OLLAMA_LOCAL
    WORKER3 --> OLLAMA_LOCAL
    
    WORKER1 --> AI_CLOUD
    WORKER2 --> AI_CLOUD
    WORKER3 --> AI_CLOUD
    
    %% Database Replication
    POSTGRES_PRIMARY --> POSTGRES_REPLICA
    
    %% Monitoring Connections
    API1 --> PROMETHEUS
    API2 --> PROMETHEUS
    API3 --> PROMETHEUS
    WORKER1 --> PROMETHEUS
    WORKER2 --> PROMETHEUS
    WORKER3 --> PROMETHEUS
    WORKER4 --> PROMETHEUS
    
    PROMETHEUS --> GRAFANA
    PROMETHEUS --> ALERTMANAGER
    
    %% Styling
    classDef users fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef loadbalancer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef application fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef worker fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef queue fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef database fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef monitoring fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef external fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    
    class USERS users
    class LB,CDN loadbalancer
    class API1,API2,API3,WEB1,WEB2 application
    class WORKER1,WORKER2,WORKER3,WORKER4 worker
    class REDIS_QUEUE,REDIS_CACHE queue
    class POSTGRES_PRIMARY,POSTGRES_REPLICA database
    class PROMETHEUS,GRAFANA,ALERTMANAGER monitoring
    class OLLAMA_LOCAL,AI_CLOUD external